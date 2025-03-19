#ifndef DISTANCEFUNCTIONS_H
#define DISTANCEFUNCTIONS_H

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#ifdef _OPENMP
  #include <omp.h>
#else
  #define omp_get_num_threads()  1
  #define omp_get_thread_num()   0
  #define omp_get_max_threads()  1
  #define omp_get_thread_limit() 1
  #define omp_get_num_procs()    1
#endif
// [[Rcpp::plugins(openmp)]]
#include <cmath>

double haversine_dist(double lat1, double lat2, double lon1, double lon2);
double euclidean_dist(double lat1, double lat2, double lon1, double lon2);
unsigned int haversine_dist_r(double lat1, double lat2, double lon1, double lon2);
unsigned int euclidean_dist_r(double lat1, double lat2, double lon1, double lon2);

#endif



