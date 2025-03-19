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
#include "lp_filling.h"

// Function overview
// 1 lp_filling_d_d Filling of spatial sandwich in logit and probit case using dense bartlett kernel distance matrix in doubles
// 2 lp_filling_d_f Filling of spatial sandwich in logit and probit case using dense bartlett kernel distance matrix in floats
// 3 lp_filling_d_s Filling of spatial sandwich in logit and probit case using dense uniform kernel distance matrix
// 4 lp_filling_s_d Filling of spatial sandwich in logit and probit case using sparse bartlett kernel distance matrix in doubles
// 5 lp_filling_s_f Filling of spatial sandwich in logit and probit case using sparse bartlett kernel distance matrix in floats
// 6 lp_filling_s_s Filling of spatial sandwich in logit and probit case using sparse uniform kernel distance matrix

// 1 Filling of spatial sandwich in logit and probit case using dense bartlett kernel distance matrix in doubles
arma::mat lp_filling_d_d(arma::mat &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_vars, unsigned int n_cores) {
  arma::mat f(n_vars, n_vars, arma::fill::zeros);
  if(n_cores > 1) {
    // Loop over observations
    #pragma omp parallel for num_threads(n_cores)
    for(unsigned int i = 0; i < n_cores; i++) {
      arma::mat f_i(n_vars, n_vars, arma::fill::zeros);
      arma::vec hf(n_obs);
      arma::mat cf(n_vars, n_vars, arma::fill::zeros);
      for(unsigned int j = i; j < n_obs; j += n_cores) {
        // Loop over variables
        for(unsigned int k = 0; k < n_vars; k++) {
          hf = X(j,k) * e * e(j) % distances.col(j);
          cf.row(k) = hf.t() * X;
        }
        f_i += cf;
      }
      #pragma omp critical
      f += f_i;
    }
  } else {
    arma::mat cf(n_vars, n_vars, arma::fill::zeros);
    arma::vec hf(n_obs);
    // Loop over observations
    for(unsigned int i {0}; i < n_obs; i++) {
      // Loop over variables
      for(unsigned int j {0}; j < n_vars; j++) {
        hf = X(i,j) * e * e(i) % distances.col(i);
        cf.row(j) = hf.t() * X;
      }
      f += cf;
    }
  }
  return f;
}

// 2 Filling of spatial sandwich in logit and probit case using dense bartlett kernel distance matrix in floats
arma::mat lp_filling_d_f(arma::fmat &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_vars, unsigned int n_cores) {
  arma::mat f(n_vars, n_vars, arma::fill::zeros);
  if(n_cores > 1) {
    // Loop over observations
    #pragma omp parallel for num_threads(n_cores)
    for(unsigned int i = 0; i < n_cores; i++) {
      arma::mat f_i(n_vars, n_vars, arma::fill::zeros);
      arma::vec hf(n_obs);
      arma::mat cf(n_vars, n_vars, arma::fill::zeros);
      for(unsigned int j = i; j < n_obs; j += n_cores) {
        // Loop over variables
        for(unsigned int k = 0; k < n_vars; k++) {
          hf = X(j,k) * e * e(j) % distances.col(j);
          cf.row(k) = hf.t() * X;
        }
        f_i += cf;
      }
      #pragma omp critical
      f += f_i;
    }
  } else {
    arma::vec hf(n_obs);
    arma::mat cf(n_vars, n_vars, arma::fill::zeros);
    // Loop over observations
    for(unsigned int i {0}; i < n_obs; i++) {
      // Loop over variables
      for(unsigned int j {0}; j < n_vars; j++) {
        hf = X(i,j) * e * e(i) % distances.col(i);
        cf.row(j) = hf.t() * X;
      }
      f += cf;
    }
  }
  return f;
}

// 3 Filling of spatial sandwich in logit and probit case using dense uniform kernel distance matrix
arma::mat lp_filling_d_s(arma::Mat<short> &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_vars, unsigned int n_cores) {
  arma::mat f(n_vars, n_vars, arma::fill::zeros);
  if(n_cores > 1) {
    // Loop over observations
    #pragma omp parallel for num_threads(n_cores)
    for(unsigned int i = 0; i < n_cores; i++) {
      arma::mat f_i(n_vars, n_vars, arma::fill::zeros);
      arma::vec hf(n_obs);
      arma::mat cf(n_vars, n_vars, arma::fill::zeros);
      for(unsigned int j = i; j < n_obs; j += n_cores) {
        // Loop over variables
        for(unsigned int k = 0; k < n_vars; k++) {
          hf = X(j,k) * e * e(j) % distances.col(j);
          cf.row(k) = hf.t() * X;
        }
        f_i += cf;
      }
      #pragma omp critical
      f += f_i;
    }
  } else {
    arma::vec hf(n_obs);
    arma::mat cf(n_vars, n_vars, arma::fill::zeros);
    // Loop over observations
    for(unsigned int i {0}; i < n_obs; i++) {
      // Loop over variables
      for(unsigned int j {0}; j < n_vars; j++) {
        hf = X(i,j) * e * e(i) % distances.col(i);
        cf.row(j) = hf.t() * X;
      }
      f += cf;
    }
  }
  return f;
}

// 4 Filling of spatial sandwich in logit and probit case using sparse bartlett kernel distance matrix in doubles
arma::mat lp_filling_s_d(arma::sp_mat &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_vars, unsigned int n_cores) {
  arma::mat f(n_vars, n_vars, arma::fill::zeros);
  if(n_cores > 1) {
    // Loop over observations
    #pragma omp parallel for num_threads(n_cores)
    for(unsigned int i = 0; i < n_cores; i++) {
      arma::mat f_i(n_vars, n_vars, arma::fill::zeros);
      arma::vec hf(n_obs);
      arma::mat cf(n_vars, n_vars, arma::fill::zeros);
      for(unsigned int j = i; j < n_obs; j += n_cores) {
        // Loop over variables
        for(unsigned int k = 0; k < n_vars; k++) {
          hf = X(j,k) * e * e(j) % distances.col(j);
          cf.row(k) = hf.t() * X;
        }
        f_i += cf;
      }
      #pragma omp critical
      f += f_i;
    }
  } else {
    arma::vec hf(n_obs);
    arma::mat cf(n_vars, n_vars, arma::fill::zeros);
    // Loop over observations
    for(unsigned int i {0}; i < n_obs; i++) {
      // Loop over variables
      for(unsigned int j {0}; j < n_vars; j++) {
        hf = X(i,j) * e * e(i) % distances.col(i);
        cf.row(j) = hf.t() * X;
      }
      f += cf;
    }
  }
  return f;
}

// 5 Filling of spatial sandwich in logit and probit case using sparse bartlett kernel distance matrix in floats
arma::mat lp_filling_s_f(arma::sp_fmat &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_vars, unsigned int n_cores) {
  arma::mat f(n_vars, n_vars, arma::fill::zeros);
  if(n_cores > 1) {
    // Loop over observations
    #pragma omp parallel for num_threads(n_cores)
    for(unsigned int i = 0; i < n_cores; i++) {
      arma::mat f_i(n_vars, n_vars, arma::fill::zeros);
      arma::vec hf(n_obs);
      arma::mat cf(n_vars, n_vars, arma::fill::zeros);
      for(unsigned int j = i; j < n_obs; j += n_cores) {
        // Loop over variables
        for(unsigned int k = 0; k < n_vars; k++) {
          hf = X(j,k) * e * e(j) % distances.col(j);
          cf.row(k) = hf.t() * X;
        }
        f_i += cf;
      }
      #pragma omp critical
      f += f_i;
    }
  } else {
    arma::vec hf(n_obs);
    arma::mat cf(n_vars, n_vars, arma::fill::zeros);
    // Loop over observations
    for(unsigned int i {0}; i < n_obs; i++) {
      // Loop over variables
      for(unsigned int j {0}; j < n_vars; j++) {
        hf = X(i,j) * e * e(i) % distances.col(i);
        cf.row(j) = hf.t() * X;
      }
      f += cf;
    }
  }
  return f;
}

// 6 Filling of spatial sandwich in logit and probit case using sparse uniform kernel distance matrix
arma::mat lp_filling_s_s(arma::SpMat<short> &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_vars, unsigned int n_cores) {
  arma::mat f(n_vars, n_vars, arma::fill::zeros);
  if(n_cores > 1) {
    // Loop over observations
    #pragma omp parallel for num_threads(n_cores)
    for(unsigned int i = 0; i < n_cores; i++) {
      arma::mat f_i(n_vars, n_vars, arma::fill::zeros);
      arma::vec hf(n_obs);
      arma::mat cf(n_vars, n_vars, arma::fill::zeros);
      for(unsigned int j = i; j < n_obs; j += n_cores) {
        // Loop over variables
        for(unsigned int k = 0; k < n_vars; k++) {
          hf = X(j,k) * e * e(j) % distances.col(j);
          cf.row(k) = hf.t() * X;
        }
        f_i += cf;
      }
      #pragma omp critical
      f += f_i;
    }
  } else {
    arma::vec hf(n_obs);
    arma::mat cf(n_vars, n_vars, arma::fill::zeros);
    // Loop over observations
    for(unsigned int i {0}; i < n_obs; i++) {
      // Loop over variables
      for(unsigned int j {0}; j < n_vars; j++) {
        hf = X(i,j) * e * e(i) % distances.col(i);
        cf.row(j) = hf.t() * X;
      }
      f += cf;
    }
  }
  return f;
}

