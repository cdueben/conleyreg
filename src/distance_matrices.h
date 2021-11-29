#ifndef DISTANCEMATRICES_H
#define DISTANCEMATRICES_H

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
#include <omp.h>
// [[Rcpp::plugins(openmp)]]
#include <cmath>

void dist_mat_d(arma::mat &distances, arma::mat &coords, unsigned int n_obs_t, double dist_cutoff, bool haversine, unsigned int n_cores);
void dist_mat_f(arma::fmat &distances, arma::mat &coords, unsigned int n_obs_t, double dist_cutoff, bool haversine, unsigned int n_cores);
void dist_mat_s(arma::Mat<short> &distances, arma::mat &coords, unsigned int n_obs_t, double dist_cutoff, bool haversine, unsigned int n_cores);
void dist_spmat_d(arma::sp_mat &distances, arma::mat &coords, unsigned int n_obs_t, double dist_cutoff, bool haversine, unsigned int n_cores);
void dist_spmat_f(arma::sp_fmat &distances, arma::mat &coords, unsigned int n_obs_t, double dist_cutoff, bool haversine, unsigned int n_cores);
void dist_spmat_s(arma::SpMat<short> &distances, arma::mat &coords, unsigned int n_obs_t, double dist_cutoff, bool haversine, unsigned int n_cores);

#endif

