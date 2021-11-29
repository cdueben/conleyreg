#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
#include <omp.h>
// [[Rcpp::plugins(openmp)]]
#include <cmath>
#include "distance_functions.h"

// 1a Function computing haversine distance
double haversine_dist(double lat1, double lat2, double lon1, double lon2) {
  double the_terms = (std::pow(std::sin((lat2 - lat1) / 2), 2)) + (std::cos(lat1) * std::cos(lat2) * std::pow(std::sin((lon2 - lon1) / 2), 2));
  double delta_sigma = 2 * std::atan2(std::sqrt(the_terms), std::sqrt(1 - the_terms));
  return (6371.01 * delta_sigma);
}

// 1b Function computing euclidean distance
double euclidean_dist(double lat1, double lat2, double lon1, double lon2) {
  return (std::sqrt(std::pow(lon1 - lon2, 2) + std::pow(lat1 - lat2, 2)));
}

// 1c Function computing rounded haversine distance
unsigned int haversine_dist_r(double lat1, double lat2, double lon1, double lon2) {
  double the_terms = (std::pow(std::sin((lat2 - lat1) / 2), 2)) + (std::cos(lat1) * std::cos(lat2) * std::pow(std::sin((lon2 - lon1) / 2), 2));
  double delta_sigma = 2 * std::atan2(std::sqrt(the_terms), std::sqrt(1 - the_terms));
  unsigned int dist_r = (int)(6371.01 * delta_sigma + 0.5);
  return dist_r;
}

// 1d Function computing rounded euclidean distance
unsigned int euclidean_dist_r(double lat1, double lat2, double lon1, double lon2) {
  unsigned int dist_r =  (int)(std::sqrt(std::pow(lon1 - lon2, 2) + std::pow(lat1 - lat2, 2)));
  return dist_r;
}


