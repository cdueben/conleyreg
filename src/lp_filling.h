#ifndef LPFILLING_H
#define LPFILLING_H

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

arma::mat lp_filling_d_d(arma::mat &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_vars, unsigned int n_cores);
arma::mat lp_filling_d_f(arma::fmat &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_vars, unsigned int n_cores);
arma::mat lp_filling_d_s(arma::Mat<short> &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_vars, unsigned int n_cores);
arma::mat lp_filling_s_d(arma::sp_mat &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_vars, unsigned int n_cores);
arma::mat lp_filling_s_f(arma::sp_fmat &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_vars, unsigned int n_cores);
arma::mat lp_filling_s_s(arma::SpMat<short> &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_vars, unsigned int n_cores);

#endif
