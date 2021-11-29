#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
#include <omp.h>
// [[Rcpp::plugins(openmp)]]
#include <cmath>
#include "XeeXhC.h"

// Function overview
// 1 XeeXhC_d_d_R Spatial sandwich using dense bartlett kernel distance matrix
// 2 XeeXhC_d_s_R Spatial sandwich using dense uniform kernel distance matrix
// 3 XeeXhC_s_d_R Spatial sandwich using sparse bartlett kernel distance matrix (R does not export to arma::SpMat<short> objects in C++; thus bartlett and uniform use same type)

// 1 Spatial sandwich using dense bartlett kernel distance matrix
// [[Rcpp::export]]
arma::mat XeeXhC_d_d_R(arma::mat &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_obs_t, unsigned int n_vars, unsigned int n_cores) {
  arma::mat XeeXh = XeeXhC_d_d(distances, X, e, n_obs, n_obs_t, n_vars, n_cores);
  return XeeXh;
}

// 2 Spatial sandwich using dense uniform kernel distance matrix
// [[Rcpp::export]]
arma::mat XeeXhC_d_s_R(arma::Mat<short> &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_obs_t, unsigned int n_vars, unsigned int n_cores) {
  arma::mat XeeXh = XeeXhC_d_s(distances, X, e, n_obs, n_obs_t, n_vars, n_cores);
  return XeeXh;
}

// 3 Spatial sandwich using sparse bartlett kernel distance matrix
// [[Rcpp::export]]
arma::mat XeeXhC_s_d_R(arma::sp_mat &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_obs_t, unsigned int n_vars, unsigned int n_cores) {
  arma::mat XeeXh = XeeXhC_s_d(distances, X, e, n_obs, n_obs_t, n_vars, n_cores);
  return XeeXh;
}

