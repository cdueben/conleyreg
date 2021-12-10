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
#include "lp_filling.h"

// 1 lp_filling_d_d_R Filling of spatial sandwich in logit and probit case using dense bartlett kernel distance matrix in doubles
// [[Rcpp::export]]
arma::mat lp_filling_d_d_R(arma::mat &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_vars, unsigned int n_cores) {
  return lp_filling_d_d(distances, X, e, n_obs, n_vars, n_cores);
}

// 2 lp_filling_d_s_R Filling of spatial sandwich in logit and probit case using dense uniform kernel distance matrix
// [[Rcpp::export]]
arma::mat lp_filling_d_s_R(arma::Mat<short> &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_vars, unsigned int n_cores) {
  return lp_filling_d_s(distances, X, e, n_obs, n_vars, n_cores);
}

// 3 lp_filling_s_d_R Filling of spatial sandwich in logit and probit case using sparse bartlett kernel distance matrix in doubles
// [[Rcpp::export]]
arma::mat lp_filling_s_d_R(arma::sp_mat &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_vars, unsigned int n_cores) {
  return lp_filling_s_d(distances, X, e, n_obs, n_vars, n_cores);
}





