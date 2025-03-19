#ifndef XEEXHCFUNCTIONS_H
#define XEEXHCFUNCTIONS_H

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

arma::mat XeeXhC_d_d(arma::mat &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_obs_t, unsigned int n_vars, unsigned int n_cores);
arma::mat XeeXhC_d_f(arma::fmat &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_obs_t, unsigned int n_vars, unsigned int n_cores);
arma::mat XeeXhC_d_s(arma::Mat<short> &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_obs_t, unsigned int n_vars, unsigned int n_cores);
arma::mat XeeXhC_s_d(arma::sp_mat &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_obs_t, unsigned int n_vars, unsigned int n_cores);
arma::mat XeeXhC_s_f(arma::sp_fmat &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_obs_t, unsigned int n_vars, unsigned int n_cores);
arma::mat XeeXhC_s_s(arma::SpMat<short> &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_obs_t, unsigned int n_vars, unsigned int n_cores);

#endif
