#
# This is a CMake makefile.  You can find the cmake utility and
# information about it at http://www.cmake.org
#
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11 -pthread")
set(CMAKE_CXX_FLAGS_DEBUG          "-g")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
set(CMAKE_CXX_COMPILER             "/usr/bin/g++-7")

# adapt to the directory where you placed
# https://github.com/imneme/pcg-cpp/blob/master/include/pcg_random.hpp
# http://www.midaco-solver.com/data/gtopx/cpp/gtopx.cpp
# and the Eigen include files (requires Eigen version >= 3.3.90)

INCLUDE_DIRECTORIES(/home/xxx/git/smart-retry/include)

if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
      RelWithDebInfo MinSizeRel." FORCE)
endif()

PROJECT(smart_retry)

add_executable(smart_retry smartretry.cpp acmasoptimizer.cpp deoptimizer.cpp)


 
