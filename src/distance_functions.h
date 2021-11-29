#ifndef DISTANCEFUNCTIONS_H
#define DISTANCEFUNCTIONS_H

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
#include <omp.h>
// [[Rcpp::plugins(openmp)]]
#include <cmath>

double haversine_dist(double lat1, double lat2, double lon1, double lon2);
double euclidean_dist(double lat1, double lat2, double lon1, double lon2);
unsigned int haversine_dist_r(double lat1, double lat2, double lon1, double lon2);
unsigned int euclidean_dist_r(double lat1, double lat2, double lon1, double lon2);

#endif



