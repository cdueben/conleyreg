#ifndef DISTANCEMATRICES_H
#define DISTANCEMATRICES_H

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
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

void dist_mat_d(arma::mat &distances, arma::mat &coords, unsigned int n_obs_t, double dist_cutoff, bool haversine, unsigned int n_cores);
void dist_mat_f(arma::fmat &distances, arma::mat &coords, unsigned int n_obs_t, double dist_cutoff, bool haversine, unsigned int n_cores);
void dist_mat_s(arma::Mat<short> &distances, arma::mat &coords, unsigned int n_obs_t, double dist_cutoff, bool haversine, unsigned int n_cores);
void dist_spmat_d(arma::sp_mat &distances, arma::mat &coords, unsigned int n_obs_t, double dist_cutoff, bool haversine, unsigned int n_cores);
void dist_spmat_f(arma::sp_fmat &distances, arma::mat &coords, unsigned int n_obs_t, double dist_cutoff, bool haversine, unsigned int n_cores);
void dist_spmat_s(arma::SpMat<short> &distances, arma::mat &coords, unsigned int n_obs_t, double dist_cutoff, bool haversine, unsigned int n_cores);

#endif

