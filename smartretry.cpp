// Copyright (c) Dietmar Wolz.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory.

// Smart parallel retry to produce reference results for
// the http://www.midaco-solver.com/index.php/about/benchmarks/gtopx
// Space Mission Benchmark problems, see
// https://www.sciencedirect.com/science/article/pii/S235271102100011X

#include <bits/exception.h>
#include <bits/std_abs.h>
#include <algorithm>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <iterator>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
// from https://github.com/imneme/pcg-cpp/blob/master/include/pcg_random.hpp
#include <pcg_random.hpp>
// from http://www.midaco-solver.com/data/gtopx/cpp/gtopx.cpp
#include <gtopx.cpp>

using namespace std;
using Clock = std::chrono::steady_clock;
using std::chrono::time_point;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

typedef double (*callback_type)(int, double[]);

extern "C"
double* optimizeDE_C(
        long runid, callback_type fit,
        int dim, int seed,
        double *lower, double *upper, int maxEvals, double keep,
        double stopfitness, int popsize, double F, double CR);

extern "C"
double* optimizeACMAS_C(long runid, callback_type fit, int dim,
        double *init, double *lower, double *upper, double *sigma, int maxIter,
        int maxEvals, double stopfitness, int mu, int popsize, double accuracy,
        long seed, bool normalize, int update_gap);

namespace smart_retry {

typedef std::vector<double> vec;

static uniform_real_distribution<> distr_01 = std::uniform_real_distribution<>(
        0, 1);

template<typename T>
string str(T begin, T end) {
    stringstream ss;
    ss << "[";
    bool first = true;
    for (; begin != end; begin++) {
        if (!first)
            ss << ", ";
        ss << *begin;
        first = false;
    }
    ss << "]";
    return ss.str();
}

string str(const vec &x) {
    return str(x.begin(), x.end());
}

string str(const vec &x, int maxSize) {
    int num = min((int)x.size(), maxSize);
    return str(x.begin(), x.begin() + num);
}

double norm(vec x) {
    double sum = 0;
    for (int i = 0; i < x.size(); i++)
        sum += x[i] * x[i];
    return sqrt(sum);
}

vec vec_x(int size, double x) {
    vec m;
    for (int i = 0; i < size; i++)
        m.push_back(x);
    return m;
}

vec add(vec x1, vec x2) {
    vec m;
    for (int i = 0; i < x1.size(); i++)
        m.push_back(x1[i] + x2[i]);
    return m;
}

vec subtract(vec x1, vec x2) {
    vec m;
    for (int i = 0; i < x1.size(); i++)
        m.push_back(x1[i] - x2[i]);
    return m;
}

vec subtractAbs(vec x1, vec x2) {
    vec m;
    for (int i = 0; i < x1.size(); i++)
        m.push_back(abs(x1[i] - x2[i]));
    return m;
}

vec divide(vec x1, vec x2) {
    vec m;
    for (int i = 0; i < x1.size(); i++)
        m.push_back(x1[i] / x2[i]);
    return m;
}

vec multiplyAbs(vec x, double f) {
    vec m;
    for (int i = 0; i < x.size(); i++)
        m.push_back(abs(x[i] * f));
    return m;
}

vec maximum(const vec& x, double d) {
    vec m;
    for (int i = 0; i < x.size(); i++)
        m.push_back(max(x[i], d));
    return m;
}

vec maximum(const vec& x1, const vec& x2) {
    vec m;
    for (int i = 0; i < x1.size(); i++)
        m.push_back(max(x1[i], x2[i]));
    return m;
}

vec minimum(const vec& x, double d) {
    vec m;
    for (int i = 0; i < x.size(); i++)
        m.push_back(min(x[i], d));
    return m;
}

vec minimum(const vec& x1, const vec& x2) {
    vec m;
    for (int i = 0; i < x1.size(); i++)
        m.push_back(min(x1[i], x2[i]));
    return m;
}

vec fitting(const vec& x, double minv, double maxv) {
    vec m;
    for (int i = 0; i < x.size(); i++)
        m.push_back(min(max(x[i], minv), maxv));
    return m;
}

vec fitting(const vec& x, const vec minx,
        const vec maxx) {
    vec m;
    for (int i = 0; i < x.size(); i++)
        m.push_back(min(max(x[i], minx[i]), maxx[i]));
    return m;
}

struct opt_result {
public:

    void set(double y, const vec& x) {
        y_ = y;
        x_ = x;
    }

    bool empty() {
        return x_.empty();
    }

    double y_;
    vec x_;
};

static thread_local pcg64* rs_;

class retry_executor {

public:

    double best_y_;
    vec best_x_;
    atomic_llong count_all_;

    retry_executor(int workers, callback_type fit, vec& lower, vec& upper, int runs,
            double limit_y, double stop_y, int start_evals,
            int max_eval_fac, int check_interval,
            double de_evals_min, double de_evals_max,
            int popsize, long seed, bool log):
                workers_(workers), fit_(fit), lower_(lower), upper_(upper), runs_(runs),
                limit_y_(limit_y), stop_y_(stop_y),
                start_evals_(start_evals), max_eval_fac_(max_eval_fac),
                check_interval_(check_interval), eval_fac_(1.0),
                de_evals_min_(de_evals_min), de_evals_max_(de_evals_max), popsize_(popsize),
                num_sorted_(0), num_stored_(0), store_size_(500), best_y_(DBL_MAX),
                log_(log) {

        count_all_.store(0);
        next_.store(0);
        dim_ = lower_.size();
        delta_ = subtract(upper_, lower_);
        seed_ = seed;
        t0_ = Clock::now();
        eval_fac_incr_ = max_eval_fac / (((double)runs) / check_interval);
        if (workers_ <= 0)
            workers_ = std::thread::hardware_concurrency();
        for (int thread_id = 0; thread_id < workers_; thread_id++) {
            jobs_.push_back(retry_job(thread_id, this));
        }
    }

    ~retry_executor() {
        for (auto &job : jobs_) {
            job.join();
        }
        delete rs_;
    }

    int millies() {
        time_point<Clock> end = Clock::now();
        milliseconds diff = duration_cast<milliseconds>(end - t0_);
        return diff.count();
    }

    double rnd01() {
        return distr_01(*rs_);
    }

    bool rnd(double probability) {
        return rnd01() < probability;
    }

    double rnd(double min, double max) {
        return min + distr_01(*rs_) * (max - min);
    }

    int rnd_int(int max) {
        return (int) (max * distr_01(*rs_));
    }

    vec rnd_x() {
        vec x;
        for (int i = 0; i < dim_; i++)
            x.push_back(rnd(lower_[i], upper_[i]));
        return x;
    }

    vec rnd_vec(double min, double max) {
        vec v;
        for (int i = 0; i < dim_; i++)
            v.push_back(rnd(min, max));
        return v;
    }

    bool crossover(int thread_id, int retry_id) {
        if (rnd(0.5) || num_sorted_ < 2)
             return false;
        int i1 = -1;
        int i2 = -1;
        int n = num_sorted_;
        double prob = rnd(min(0.1 * n, 1.0), 0.2 * n) / n;
        bool stop = false;
        for (int i = 0; i < 100 && !stop; i++) {
            i1 = -1;
            i2 = -1;
            for (int j = 0; j < n; j++) {
                if (rnd(prob)) {
                    if (i1 < 0)
                        i1 = j;
                    else {
                        i2 = j;
                        stop = true;
                        break;
                    }
                }
            }
        }
        if (i2 < 0)
            return false;
        opt_result opt_res1 = store_[i1];
        opt_result opt_res2 = store_[i2];
        vec x0 = opt_res1.x_;
        vec x1 = opt_res2.x_;

        double diffFac = rnd(0.5, 1.0);
        double limFac = rnd(2.0, 4.0) * diffFac;

        vec deltax = subtractAbs(x1, x0);
        vec delta_bound = maximum(multiplyAbs(deltax, limFac),
                0.0001);
        vec lower = maximum(lower_, subtract(x0, delta_bound));
        vec upper = minimum(upper_, add(x0, delta_bound));
        vec guess = fitting(x1, lower, upper);
        vec stepsize = fitting(multiplyAbs(divide(deltax, delta_), diffFac), 0.001, 0.5);
        optimize(thread_id, retry_id, opt_res1.y_, lower, upper,
                    guess, stepsize);
        return true;
    }

    int eval_num() {
        return (int) (eval_fac_ * start_evals_);
    }

    void incr_count_evals(int retry_id, int evals) {
        if (retry_id % check_interval_ == check_interval_ - 1) {
            if (eval_fac_ < max_eval_fac_)
                eval_fac_ += eval_fac_incr_;
            if (num_stored_ > 1)
                sort();
        }
        count_all_.fetch_add(evals);
    }

    double distance(const opt_result& or1, const opt_result& or2) {
        return norm(divide(subtract(or1.x_, or2.x_), delta_)) / sqrt(dim_);
    }

    string store_str(int size) {
        vec vals;
        for (int i = 0; i < size && i < store_.size(); i++)
            vals.push_back(store_[i].y_);
        return str(vals);
    }

    void sort() {
        vector<int> sorted;
        for (int i = 0; i < num_stored_; i++)
            sorted.push_back(i);
        std::sort(sorted.begin(), sorted.end(),
                [&]( int i1, int i2 ) {
                   return store_[i1].y_ < store_[i2].y_;
                });
        int prev = -1;
        int prev2 = -1;
        vector<opt_result> store(store_);
        store_.clear();
        for (int i = 0; i < num_stored_; i++) {
            int opt_res = sorted[i];
            if ((prev < 0 || distance(store[prev], store[opt_res]) > 0.15)
                    && (prev2 < 0 || distance(store[prev2], store[opt_res]) > 0.15)) {
                store_.push_back(store[opt_res]);
                prev2 = prev;
                prev = opt_res;
            }
        }
        num_sorted_ = num_stored_ = min((int) store_.size(), (int) (0.9 * store_size_));
    }

    void dump(int retry_id) {
        if (log_ && num_stored_ > 0) {
            long evals = count_all_.load();
            int ms = millies();
            long evals_per_sec = evals*1000 / ms;
            double best = best_y_;
            double worst = store_[num_stored_ - 1].y_;

            cout << retry_id << " " << num_stored_ << " " << eval_fac_ << " " << ms/1000.0 << " "
                << evals << " " << evals_per_sec << " " << best << " " << worst << " n="
                << stat_y_.i << " m=" << stat_y_.ai << " s=" << stat_y_.sdev() << " "
                << store_str(20) << " " << str(best_x_) << endl << flush;
        }
    }

    void add_result(int retry_id, int evals, double bestY,
                const vec& bestX, double limit_y) {
        unique_lock<mutex> lock(store_mutex_);
        incr_count_evals(retry_id, evals);
        if (bestY < limit_y) {
            stat_y_.add(bestY);
            if (bestY < best_y_) {
                best_y_ = bestY;
                best_x_ = bestX;
                dump(retry_id);
            }
            if (num_stored_ < store_size_) {
                if (store_.size() <= num_stored_)
                    store_.push_back(opt_result());
            } else {
                sort();
//              dump(retry_id);
            }
            store_[num_stored_++].set(bestY, bestX);
        }
        lock.unlock();
    }

    void optimize(int id, int retry_id, double limit_y,
            vec &lower, vec &upper,
            vec &guess, vec &stepsize) {

        double low[dim_];
        double up[dim_];
        double init[dim_];
        double sdev[dim_];
        for (int i = 0; i < dim_; i++) {
            low[i] = lower[i];
            up[i] = upper[i];
            init[i] = guess[i];
            sdev[i] = stepsize[i];
        }
        int en = eval_num();

        double de_evals = rnd(de_evals_min_, de_evals_max_);
        double cma_evals = 1.0 - de_evals;
        double* res_de = optimizeDE_C(
                retry_id, fit_, dim_,
                rnd_int(INT_MAX), low, up, (int) (de_evals * en), 200,
                stop_y_, popsize_, 0.5, 0.9);

        double best_y_de = res_de[dim_];
        int evals = int(res_de[dim_+1]);
        double* res = optimizeACMAS_C(retry_id, fit_, dim_,
                res_de, low, up, sdev, 10000000, (int) (cma_evals * en),
                stop_y_, popsize_/2, popsize_, 1.0,
                rnd_int(INT_MAX), true, -1);
        double best_y = res[dim_];
        evals += int(res[dim_+1]);
        if (best_y_de <= best_y) {
            best_y = best_y_de;
            delete[] res;
            res = res_de;
        } else
            delete[] res_de;
        vec best_x;
        for (int i = 0; i < dim_; i++)
            best_x.push_back(res[i]);
        delete[] res;
        add_result(retry_id, evals, best_y, best_x, limit_y);
    }

    void execute(int thread_id) {
        rs_ = new pcg64(clock() + 67*thread_id + seed_);
        int retry_id;
        while ((retry_id = next_.fetch_add(1)) < runs_
                && best_y_ >= stop_y_) {
            if (crossover(thread_id, retry_id))
                continue;
            int evals = eval_num();
            vec guess = rnd_x();
            vec stepsize = vec_x(dim_, rnd(0.05, 0.1));
            optimize(thread_id, retry_id, limit_y_,
                        lower_, upper_,
                        guess, stepsize);
        }
    }

    void join() {
        for (auto &job : jobs_) {
            job.join();
        }
        dump(next_.load());
    }

private:

    class retry_job {

    public:

        retry_job(int id, retry_executor *exec) {
            thread_ = thread(&retry_executor::execute, exec, id);
        }

        void join() {
            if (thread_.joinable())
                thread_.join();
        }

    private:
        thread thread_;
    };

    class statistics {

    public:

        double ai = 0;
        double qi = 0;
        int i = 0;

        void add(double value) {
            i++;
            if (i == 1)
                ai = value;
            else {
                qi += (i - 1) * (value - ai) * (value - ai) / i;
                ai += (value - ai) / i;
            }
        }

        double sdev() {
            return i == 0 ? 0 : sqrt(qi / i);
                return 0;
        }
    };

    int workers_;
    int dim_;
    callback_type fit_;
    int runs_;
    vec lower_;
    vec upper_;
    double limit_y_;
    double stop_y_;
    double start_evals_;
    int max_eval_fac_;
    double eval_fac_incr_;
    double eval_fac_;
    int check_interval_;
    double de_evals_min_;
    double de_evals_max_;
    int popsize_;
    vector<retry_job> jobs_;
    vector<opt_result> store_;
    mutex store_mutex_;
    atomic_long next_;
    int num_stored_;
    int num_sorted_;
    int store_size_;
    vec delta_;
    long seed_;
    bool log_;
    statistics stat_y_;
    time_point<Clock> t0_;
};
}

double rosen(int n, double* x) {
    double f = 0;
    for (int i = 0; i < n - 1; i++)
        f += 1e2 * (x[i] * x[i] - x[i + 1]) * (x[i] * x[i] - x[i + 1]) + (x[i] - 1.) * (x[i] - 1.);
    return f;
}

std::vector<double> getVector(int n, double *x) {
    std::vector<double> X;
    for (int i = 0; i < n; i++)
        X.push_back(x[i]);
    return X;
}

double gtoc1_obj(int n, double *x) {
    std::vector<double> rp;
    double val = gtoc1(getVector(n, x), rp);
    rp.clear();
    return val - 2000000;
}

double cassini1_obj(int n, double *x) {
    std::vector<double> rp;
    double launchDV;
    double val = cassini1(getVector(n, x), rp, launchDV);
    rp.clear();
    return val;
}

double messenger_obj(int n, double *x) {
    double launchDV, flightDV, arrivalDV;
    return messenger(getVector(n, x), launchDV, flightDV, arrivalDV);
}

double messengerfull_obj(int n, double *x) {
    double launchDV, flightDV, arrivalDV;
    return messengerfull(getVector(n, x), launchDV, flightDV, arrivalDV);
}

double cassini2_obj(int n, double *x) {
    double launchDV, flightDV, arrivalDV;
    return cassini2(getVector(n, x), launchDV, flightDV, arrivalDV);
}

double rosetta_obj(int n, double *x) {
    double launchDV, flightDV, arrivalDV;
    return rosetta(getVector(n, x), launchDV, flightDV, arrivalDV);
}

double sagas_obj(int n, double *x) {
    double DVtot, DVonboard;
    double obj = sagas(getVector(n, x), DVtot, DVonboard);
    // add constraint penalty
    if (DVtot > 6.782)
         obj += 10+10*DVtot;
    if (DVonboard > 1.782)
         obj += 10+10*DVonboard;
    return obj;
}

double tandem_obj(int n, double *x) {
    double tof = 0;
    int seq[5] = { 3, 2, 3, 3, 6 };
    double val = -tandem(getVector(n, x), tof, seq);
    // add constraint penalty
    if (tof > 3652.5)
        val += 1000 + 1000 * (tof - 3652.5);
    return val;
}

using namespace smart_retry;

double optimizeSmart(callback_type fit, int dim,
        double *lower, double *upper, int start_evals,
        int max_eval_fac, int check_interval,
        double de_evals_min, double de_evals_max, double limit_y,
        double stop_y, int popsize, long seed, int workers, int runs, bool log) {
    int n = dim;
    vec lower_lim(lower, lower+n);
    vec upper_lim(upper, upper+n);

    if (popsize <= 0)
        popsize = 31;
    if (start_evals <= 0)
        start_evals = 1500;
    if (max_eval_fac <= 0)
        max_eval_fac = 50;
    if (check_interval <= 0)
        check_interval = 100;
    if (de_evals_min < 0)
        de_evals_min = 0.1;
    if (de_evals_max < 0)
        de_evals_max = 0.3;
    if (runs <= 0)
        runs = 10000;
    try {
        smart_retry::retry_executor exec(workers, fit, lower_lim, upper_lim, runs,
                    limit_y, stop_y, start_evals, max_eval_fac, check_interval,
                    de_evals_min, de_evals_max, popsize, seed, log);
        exec.join();
        vec bestX = exec.best_x_;
        double bestY = exec.best_y_;
        long evals = exec.count_all_.load();
        return bestY;
     } catch (std::exception &e) {
        cout << e.what() << endl;
        return 0;;
    }
}

struct problem {
public:

    problem(callback_type fit, string name, int dim, double* lower, double* upper,
            double limit_y, double stop_y, int runs, int popsize) :
                fit_(fit), name_(name), dim_(dim),
                lower_(lower), upper_(upper), limit_y_(limit_y),
                stop_y_(stop_y), runs_(runs), popsize_(popsize) {
    }

    double optimize() {
        long seed = rand();
        return optimizeSmart(fit_, dim_, lower_, upper_, 0,
                0, 0, 0.1, 0.5, limit_y_, stop_y_, popsize_, seed, 0, runs_, true);
    }

    callback_type fit_;
    string name_;
    const int dim_;
    double* lower_;
    double* upper_;
    double limit_y_;
    double stop_y_;
    int runs_;
    int popsize_;

};

static vector<problem> problems() {
    vector<problem> probs;

    probs.push_back(problem(gtoc1_obj, "gtoc1", 8,
                    new double[8]{ 3000., 14., 14., 14., 14., 100., 366., 300. },
                    new double[8]{ 10000., 2000., 2000., 2000., 2000., 9000., 9000., 9000. },
                        -300000, -1581949, 10000, 31));

    probs.push_back(problem(cassini1_obj, "cassini1", 6,
                    new double[6]{ -1000.,30.,100.,30.,400.,1000. },
                    new double[6]{ 0.,400.,470.,400.,2000.,6000. },
                        20, 4.93075, 4000, 31));

    probs.push_back(problem(cassini2_obj, "cassini2", 22,
                    new double[22]{ -1000,3,0,0,100,100,30,400,800,0.01,0.01,0.01,0.01,0.01,1.05,1.05,1.15,1.7,-M_PI,-M_PI,-M_PI,-M_PI },
                    new double[22]{ 0,5,1,1,400,500,300,1600,2200,0.9,0.9,0.9,0.9,0.9,6,6,6.5,291,M_PI,M_PI,M_PI,M_PI },
                        20, 8.38305, 6000, 31));

    probs.push_back(problem(messenger_obj, "messenger", 18,
                    new double[18]{ 1000.,1.,0.,0.,200.,30.,30.,30.,0.01,0.01,0.01,0.01,1.1,1.1,1.1,-M_PI,-M_PI,-M_PI },
                    new double[18]{ 4000.,5.,1.,1.,400.,400.,400.,400.,0.99,0.99,0.99,0.99,6,6,6,M_PI,M_PI,M_PI },
                        20, 8.62995, 8000, 31));

    probs.push_back(problem(rosetta_obj, "rosetta", 22,
                    new double[22]{ 1460,3,0,0,300,150,150,300,700,0.01,0.01,0.01,0.01,0.01,1.05,1.05,1.05,1.05,-M_PI,-M_PI,-M_PI,-M_PI },
                    new double[22]{ 1825,5,1,1,500,800,800,800,1850,0.9,0.9,0.9,0.9,0.9,9,9,9,9,M_PI,M_PI,M_PI,M_PI },
                        20, 1.34335, 4000, 31));

    probs.push_back(problem(sagas_obj, "sagas", 12,
                    new double[12]{ 7000,0,0,0,50,300,0.01,0.01,1.05,8,-M_PI,-M_PI },
                    new double[12]{ 9100,7,1,1,2000,2000,0.9,0.9,7,500,M_PI,M_PI },
                        100, 18.188, 4000, 31));

    probs.push_back(problem(tandem_obj, "tandem", 18,
                    new double[18]{ 5475, 2.5, 0, 0, 20, 20, 20, 20, 0.01, 0.01, 0.01,
                        0.01, 1.05, 1.05, 1.05, -M_PI, -M_PI, -M_PI },
                    new double[18]{ 9132, 4.9, 1, 1, 2500, 2500, 2500, 2500, 0.99, 0.99, 0.99,
                        0.99, 10, 10, 10, M_PI, M_PI, M_PI },
                        -500, -1500, 20000, 31));

    probs.push_back(problem(messengerfull_obj, "messfull", 26,
                    new double[26]{ 1900.0, 3.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 0.01, 0.01, 0.01, 0.01,
                            0.01, 0.01, 1.1, 1.1, 1.05, 1.05, 1.05, -M_PI, -M_PI, -M_PI, -M_PI, -M_PI },
                    new double[26]{ 2200.0, 4.05, 1.0, 1.0, 500.0, 500.0, 500.0, 500.0, 500.0, 550.0, 0.99, 0.99, 0.99, 0.99,
                            0.99, 0.99, 6.0, 6.0, 6.0, 6.0, 6.0, M_PI, M_PI, M_PI, M_PI, M_PI },
                        12, 1.960, 50000, 31));
    return probs;
}

void benchmark() {
    srand (time(NULL));
    vector<problem> probs = problems();
    int numRuns = 100;
    for (auto &prob : probs) {
        cout << "Testing smart retry " <<  prob.name_ << endl << flush;
        for (int i = 0; i < numRuns; i++)
            prob.optimize();
    }
}

int main() {
    benchmark();
}
