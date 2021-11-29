#ifndef LPFILLING_H
#define LPFILLING_H

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
#include <omp.h>
// [[Rcpp::plugins(openmp)]]
#include <cmath>

arma::mat lp_filling_d_d(arma::mat &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_vars, unsigned int n_cores);
arma::mat lp_filling_d_f(arma::fmat &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_vars, unsigned int n_cores);
arma::mat lp_filling_d_s(arma::Mat<short> &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_vars, unsigned int n_cores);
arma::mat lp_filling_s_d(arma::sp_mat &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_vars, unsigned int n_cores);
arma::mat lp_filling_s_f(arma::sp_fmat &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_vars, unsigned int n_cores);
arma::mat lp_filling_s_s(arma::SpMat<short> &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_vars, unsigned int n_cores);

#endif
