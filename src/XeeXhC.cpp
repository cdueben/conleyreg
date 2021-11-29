#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
#include <omp.h>
// [[Rcpp::plugins(openmp)]]
#include <cmath>
#include "XeeXhC.h"

// Function overview
// 1 XeeXhC_d_d Spatial sandwich using dense bartlett kernel distance matrix in doubles
// 2 XeeXhC_d_f Spatial sandwich using dense bartlett kernel distance matrix in floats
// 3 XeeXhC_d_s Spatial sandwich using dense uniform kernel distance matrix
// 4 XeeXhC_s_d Spatial sandwich using sparse bartlett kernel distance matrix in doubles
// 5 XeeXhC_s_f Spatial sandwich using sparse bartlett kernel distance matrix in floats
// 6 XeeXhC_s_s Spatial sandwich using sparse uniform kernel distance matrix

// 1 Spatial sandwich using dense bartlett kernel distance matrix in doubles
arma::mat XeeXhC_d_d(arma::mat &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_obs_t, unsigned int n_vars, unsigned int n_cores) {
  // Set empty output matrix
  arma::mat XeeXh(n_vars, n_vars, arma::fill::zeros);
  // Generate a [N variables x 1] matrix with all values set to one
  arma::mat k_mat(n_vars, 1, arma::fill::ones);
  // Generate a [1 x obs in that period] matrix
  arma::mat d_row(1, n_obs_t, arma::fill::ones);
  if(n_obs > n_obs_t) {
    // Balanced panel
    if(n_cores > 1) {
      #pragma omp parallel for num_threads(n_cores)
      for(unsigned int i = 0; i < n_cores; i++) {
        arma::mat XeeXh_i(n_vars, n_vars, arma::fill::zeros);
        arma::mat e_mat(1, n_obs_t, arma::fill::zeros);
        unsigned int k {};
        unsigned int l {};
        for(unsigned int j = i; j < n_obs; j += n_cores) {
          k = j % n_obs_t;
          l = j - k;
          // Set values to the residual of one obsveration
          e_mat.fill(e[j]);
          // Fill the output matrix (% represents element-wise multiplication, not modulo)
          XeeXh_i += ((k_mat % X.row(j).t()) * e_mat % (k_mat * ((d_row % distances.row(k)) % e.subvec(l, (l + n_obs_t - 1)).t()))) * X.rows(l, (l + n_obs_t - 1));
        }
        #pragma omp critical
        XeeXh += XeeXh_i;
      }
    } else {
      unsigned int j {};
      unsigned int k {};
      arma::mat e_mat(1, n_obs_t, arma::fill::zeros);
      // Loop over observations
      for(unsigned int i = 0; i < n_obs; i++) {
        j = i % n_obs_t;
        k = i - j;
        // Set values to the residual of one obsveration
        e_mat.fill(e[i]);
        // Fill the output matrix (% represents element-wise multiplication, not modulo)
        XeeXh += ((k_mat % X.row(i).t()) * e_mat % (k_mat * ((d_row % distances.row(j)) % e.subvec(k, (k + n_obs_t - 1)).t()))) * X.rows(k, (k + n_obs_t - 1));
      }
    }
  } else {
    // Cross-sectional case
    if(n_cores > 1) {
      // Loop over observations
      #pragma omp parallel for num_threads(n_cores)
      for(unsigned int i = 0; i < n_cores; i++) {
        arma::mat XeeXh_i(n_vars, n_vars, arma::fill::zeros);
        arma::mat e_mat(1, n_obs, arma::fill::zeros);
        for(unsigned int j = i; j < n_obs; j += n_cores) {
          // Set values to the residual of one obsveration
          e_mat.fill(e[j]);
          // Fill the output matrix (% represents element-wise multiplication, not modulo)
          XeeXh_i += ((k_mat % X.row(j).t()) * e_mat % (k_mat * ((d_row % distances.row(j)) % e.t()))) * X;
        }
        #pragma omp critical
        XeeXh += XeeXh_i;
      }
    } else {
      arma::mat e_mat(1, n_obs, arma::fill::zeros);
      // Loop over observations
      for(unsigned int i {0}; i < n_obs; i++) {
        e_mat.fill(e[i]);
        // Fill the output matrix (% represents element-wise multiplication, not modulo)
        XeeXh += ((k_mat % X.row(i).t()) * e_mat % (k_mat * ((d_row % distances.row(i)) % e.t()))) * X;
      }
    }
  }
  return XeeXh;
}

// 2 Spatial sandwich using dense bartlett kernel distance matrix in floats
arma::mat XeeXhC_d_f(arma::fmat &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_obs_t, unsigned int n_vars, unsigned int n_cores) {
  // Set empty output matrix
  arma::mat XeeXh(n_vars, n_vars, arma::fill::zeros);
  // Generate a [N variables x 1] matrix with all values set to one
  arma::mat k_mat(n_vars, 1, arma::fill::ones);
  // Generate a [1 x obs in that period] matrix
  arma::fmat d_row(1, n_obs_t, arma::fill::ones);
  if(n_obs > n_obs_t) {
    // Balanced panel
    if(n_cores > 1) {
      #pragma omp parallel for num_threads(n_cores)
      for(unsigned int i = 0; i < n_cores; i++) {
        arma::mat XeeXh_i(n_vars, n_vars, arma::fill::zeros);
        arma::mat e_mat(1, n_obs_t, arma::fill::zeros);
        unsigned int k {};
        unsigned int l {};
        for(unsigned int j = i; j < n_obs; j += n_cores) {
          k = j % n_obs_t;
          l = j - k;
          // Set values to the residual of one obsveration
          e_mat.fill(e[j]);
          // Fill the output matrix (% represents element-wise multiplication, not modulo)
          XeeXh_i += ((k_mat % X.row(j).t()) * e_mat % (k_mat * ((d_row % distances.row(k)) % e.subvec(l, (l + n_obs_t - 1)).t()))) * X.rows(l, (l + n_obs_t - 1));
        }
        #pragma omp critical
        XeeXh += XeeXh_i;
      }
    } else {
      unsigned int j {};
      unsigned int k {};
      arma::mat e_mat(1, n_obs_t, arma::fill::zeros);
      // Loop over observations
      for(unsigned int i = 0; i < n_obs; i++) {
        j = i % n_obs_t;
        k = i - j;
        // Set values to the residual of one obsveration
        e_mat.fill(e[i]);
        // Fill the output matrix (% represents element-wise multiplication, not modulo)
        XeeXh += ((k_mat % X.row(i).t()) * e_mat % (k_mat * ((d_row % distances.row(j)) % e.subvec(k, (k + n_obs_t - 1)).t()))) * X.rows(k, (k + n_obs_t - 1));
      }
    }
  } else {
    // Cross-sectional case
    if(n_cores > 1) {
      // Loop over observations
      #pragma omp parallel for num_threads(n_cores)
      for(unsigned int i = 0; i < n_cores; i++) {
        arma::mat XeeXh_i(n_vars, n_vars, arma::fill::zeros);
        arma::mat e_mat(1, n_obs, arma::fill::zeros);
        for(unsigned int j = i; j < n_obs; j += n_cores) {
          // Set values to the residual of one obsveration
          e_mat.fill(e[j]);
          // Fill the output matrix (% represents element-wise multiplication, not modulo)
          XeeXh_i += ((k_mat % X.row(j).t()) * e_mat % (k_mat * ((d_row % distances.row(j)) % e.t()))) * X;
        }
        #pragma omp critical
        XeeXh += XeeXh_i;
      }
    } else {
      arma::mat e_mat(1, n_obs, arma::fill::zeros);
      // Loop over observations
      for(unsigned int i {0}; i < n_obs; i++) {
        e_mat.fill(e[i]);
        // Fill the output matrix (% represents element-wise multiplication, not modulo)
        XeeXh += ((k_mat % X.row(i).t()) * e_mat % (k_mat * ((d_row % distances.row(i)) % e.t()))) * X;
      }
    }
  }
  return XeeXh;
}

// 3 Spatial sandwich using dense uniform kernel distance matrix
arma::mat XeeXhC_d_s(arma::Mat<short> &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_obs_t, unsigned int n_vars, unsigned int n_cores) {
  // Set empty output matrix
  arma::mat XeeXh(n_vars, n_vars, arma::fill::zeros);
  // Generate a [N variables x 1] matrix with all values set to one
  arma::mat k_mat(n_vars, 1, arma::fill::ones);
  // Generate a [1 x obs in that period] matrix
  arma::Mat<short> d_row(1, n_obs_t, arma::fill::ones);
  if(n_obs > n_obs_t) {
    // Balanced panel
    if(n_cores > 1) {
      #pragma omp parallel for num_threads(n_cores)
      for(unsigned int i = 0; i < n_cores; i++) {
        arma::mat XeeXh_i(n_vars, n_vars, arma::fill::zeros);
        arma::mat e_mat(1, n_obs_t, arma::fill::zeros);
        unsigned int k {};
        unsigned int l {};
        for(unsigned int j = i; j < n_obs; j += n_cores) {
          k = j % n_obs_t;
          l = j - k;
          // Set values to the residual of one obsveration
          e_mat.fill(e[j]);
          // Fill the output matrix (% represents element-wise multiplication, not modulo)
          XeeXh_i += ((k_mat % X.row(j).t()) * e_mat % (k_mat * ((d_row % distances.row(k)) % e.subvec(l, (l + n_obs_t - 1)).t()))) * X.rows(l, (l + n_obs_t - 1));
        }
        #pragma omp critical
        XeeXh += XeeXh_i;
      }
    } else {
      unsigned int j {};
      unsigned int k {};
      arma::mat e_mat(1, n_obs_t, arma::fill::zeros);
      // Loop over observations
      for(unsigned int i = 0; i < n_obs; i++) {
        j = i % n_obs_t;
        k = i - j;
        // Set values to the residual of one obsveration
        e_mat.fill(e[i]);
        // Fill the output matrix (% represents element-wise multiplication, not modulo)
        XeeXh += ((k_mat % X.row(i).t()) * e_mat % (k_mat * ((d_row % distances.row(j)) % e.subvec(k, (k + n_obs_t - 1)).t()))) * X.rows(k, (k + n_obs_t - 1));
      }
    }
  } else {
    // Cross-sectional case
    if(n_cores > 1) {
      // Loop over observations
      #pragma omp parallel for num_threads(n_cores)
      for(unsigned int i = 0; i < n_cores; i++) {
        arma::mat XeeXh_i(n_vars, n_vars, arma::fill::zeros);
        arma::mat e_mat(1, n_obs, arma::fill::zeros);
        for(unsigned int j = i; j < n_obs; j += n_cores) {
          // Set values to the residual of one obsveration
          e_mat.fill(e[j]);
          // Fill the output matrix (% represents element-wise multiplication, not modulo)
          XeeXh_i += ((k_mat % X.row(j).t()) * e_mat % (k_mat * ((d_row % distances.row(j)) % e.t()))) * X;
        }
        #pragma omp critical
        XeeXh += XeeXh_i;
      }
    } else {
      arma::mat e_mat(1, n_obs, arma::fill::zeros);
      // Loop over observations
      for(unsigned int i {0}; i < n_obs; i++) {
        e_mat.fill(e[i]);
        // Fill the output matrix (% represents element-wise multiplication, not modulo)
        XeeXh += ((k_mat % X.row(i).t()) * e_mat % (k_mat * ((d_row % distances.row(i)) % e.t()))) * X;
      }
    }
  }
  return XeeXh;
}

// 4 Spatial sandwich using sparse bartlett kernel distance matrix in doubles
arma::mat XeeXhC_s_d(arma::sp_mat &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_obs_t, unsigned int n_vars, unsigned int n_cores) {
  // Set empty output matrix
  arma::mat XeeXh(n_vars, n_vars, arma::fill::zeros);
  // Generate a [N variables x 1] matrix with all values set to one
  arma::mat k_mat(n_vars, 1, arma::fill::ones);
  // Generate a [1 x obs in that period] matrix
  arma::mat d_row(1, n_obs_t, arma::fill::ones);
  if(n_obs > n_obs_t) {
    // Balanced panel
    if(n_cores > 1) {
      #pragma omp parallel for num_threads(n_cores)
      for(unsigned int i = 0; i < n_cores; i++) {
        arma::mat XeeXh_i(n_vars, n_vars, arma::fill::zeros);
        arma::mat e_mat(1, n_obs_t, arma::fill::zeros);
        unsigned int k {};
        unsigned int l {};
        for(unsigned int j = i; j < n_obs; j += n_cores) {
          k = j % n_obs_t;
          l = j - k;
          // Set values to the residual of one obsveration
          e_mat.fill(e[j]);
          // Fill the output matrix (% represents element-wise multiplication, not modulo)
          XeeXh_i += ((k_mat % X.row(j).t()) * e_mat % (k_mat * ((d_row % distances.row(k)) % e.subvec(l, (l + n_obs_t - 1)).t()))) * X.rows(l, (l + n_obs_t - 1));
        }
        #pragma omp critical
        XeeXh += XeeXh_i;
      }
    } else {
      unsigned int j {};
      unsigned int k {};
      arma::mat e_mat(1, n_obs_t, arma::fill::zeros);
      // Loop over observations
      for(unsigned int i = 0; i < n_obs; i++) {
        j = i % n_obs_t;
        k = i - j;
        // Set values to the residual of one obsveration
        e_mat.fill(e[i]);
        // Fill the output matrix (% represents element-wise multiplication, not modulo)
        XeeXh += ((k_mat % X.row(i).t()) * e_mat % (k_mat * ((d_row % distances.row(j)) % e.subvec(k, (k + n_obs_t - 1)).t()))) * X.rows(k, (k + n_obs_t - 1));
      }
    }
  } else {
    // Cross-sectional case
    if(n_cores > 1) {
      // Loop over observations
      #pragma omp parallel for num_threads(n_cores)
      for(unsigned int i = 0; i < n_cores; i++) {
        arma::mat XeeXh_i(n_vars, n_vars, arma::fill::zeros);
        arma::mat e_mat(1, n_obs, arma::fill::zeros);
        for(unsigned int j = i; j < n_obs; j += n_cores) {
          // Set values to the residual of one obsveration
          e_mat.fill(e[j]);
          // Fill the output matrix (% represents element-wise multiplication, not modulo)
          XeeXh_i += ((k_mat % X.row(j).t()) * e_mat % (k_mat * ((d_row % distances.row(j)) % e.t()))) * X;
        }
        #pragma omp critical
        XeeXh += XeeXh_i;
      }
    } else {
      arma::mat e_mat(1, n_obs, arma::fill::zeros);
      // Loop over observations
      for(unsigned int i {0}; i < n_obs; i++) {
        e_mat.fill(e[i]);
        // Fill the output matrix (% represents element-wise multiplication, not modulo)
        XeeXh += ((k_mat % X.row(i).t()) * e_mat % (k_mat * ((d_row % distances.row(i)) % e.t()))) * X;
      }
    }
  }
  return XeeXh;
}

// 5 Spatial sandwich using sparse bartlett kernel distance matrix in floats
arma::mat XeeXhC_s_f(arma::sp_fmat &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_obs_t, unsigned int n_vars, unsigned int n_cores) {
  // Set empty output matrix
  arma::mat XeeXh(n_vars, n_vars, arma::fill::zeros);
  // Generate a [N variables x 1] matrix with all values set to one
  arma::mat k_mat(n_vars, 1, arma::fill::ones);
  // Generate a [1 x obs in that period] matrix
  arma::fmat d_row(1, n_obs_t, arma::fill::ones);
  if(n_obs > n_obs_t) {
    // Balanced panel
    if(n_cores > 1) {
      #pragma omp parallel for num_threads(n_cores)
      for(unsigned int i = 0; i < n_cores; i++) {
        arma::mat XeeXh_i(n_vars, n_vars, arma::fill::zeros);
        arma::mat e_mat(1, n_obs_t, arma::fill::zeros);
        unsigned int k {};
        unsigned int l {};
        for(unsigned int j = i; j < n_obs; j += n_cores) {
          k = j % n_obs_t;
          l = j - k;
          // Set values to the residual of one obsveration
          e_mat.fill(e[j]);
          // Fill the output matrix (% represents element-wise multiplication, not modulo)
          XeeXh_i += ((k_mat % X.row(j).t()) * e_mat % (k_mat * ((d_row % distances.row(k)) % e.subvec(l, (l + n_obs_t - 1)).t()))) * X.rows(l, (l + n_obs_t - 1));
        }
        #pragma omp critical
        XeeXh += XeeXh_i;
      }
    } else {
      unsigned int j {};
      unsigned int k {};
      arma::mat e_mat(1, n_obs_t, arma::fill::zeros);
      // Loop over observations
      for(unsigned int i = 0; i < n_obs; i++) {
        j = i % n_obs_t;
        k = i - j;
        // Set values to the residual of one obsveration
        e_mat.fill(e[i]);
        // Fill the output matrix (% represents element-wise multiplication, not modulo)
        XeeXh += ((k_mat % X.row(i).t()) * e_mat % (k_mat * ((d_row % distances.row(j)) % e.subvec(k, (k + n_obs_t - 1)).t()))) * X.rows(k, (k + n_obs_t - 1));
      }
    }
  } else {
    // Cross-sectional case
    if(n_cores > 1) {
      // Loop over observations
      #pragma omp parallel for num_threads(n_cores)
      for(unsigned int i = 0; i < n_cores; i++) {
        arma::mat XeeXh_i(n_vars, n_vars, arma::fill::zeros);
        arma::mat e_mat(1, n_obs, arma::fill::zeros);
        for(unsigned int j = i; j < n_obs; j += n_cores) {
          // Set values to the residual of one obsveration
          e_mat.fill(e[j]);
          // Fill the output matrix (% represents element-wise multiplication, not modulo)
          XeeXh_i += ((k_mat % X.row(j).t()) * e_mat % (k_mat * ((d_row % distances.row(j)) % e.t()))) * X;
        }
        #pragma omp critical
        XeeXh += XeeXh_i;
      }
    } else {
      arma::mat e_mat(1, n_obs, arma::fill::zeros);
      // Loop over observations
      for(unsigned int i {0}; i < n_obs; i++) {
        e_mat.fill(e[i]);
        // Fill the output matrix (% represents element-wise multiplication, not modulo)
        XeeXh += ((k_mat % X.row(i).t()) * e_mat % (k_mat * ((d_row % distances.row(i)) % e.t()))) * X;
      }
    }
  }
  return XeeXh;
}

// 6 Spatial sandwich using sparse uniform kernel distance matrix
arma::mat XeeXhC_s_s(arma::SpMat<short> &distances, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_obs_t, unsigned int n_vars, unsigned int n_cores) {
  // Set empty output matrix
  arma::mat XeeXh(n_vars, n_vars, arma::fill::zeros);
  // Generate a [N variables x 1] matrix with all values set to one
  arma::mat k_mat(n_vars, 1, arma::fill::ones);
  // Generate a [1 x obs in that period] matrix
  arma::Mat<short> d_row(1, n_obs_t, arma::fill::ones);
  if(n_obs > n_obs_t) {
    // Balanced panel
    if(n_cores > 1) {
      #pragma omp parallel for num_threads(n_cores)
      for(unsigned int i = 0; i < n_cores; i++) {
        arma::mat XeeXh_i(n_vars, n_vars, arma::fill::zeros);
        arma::mat e_mat(1, n_obs_t, arma::fill::zeros);
        unsigned int k {};
        unsigned int l {};
        for(unsigned int j = i; j < n_obs; j += n_cores) {
          k = j % n_obs_t;
          l = j - k;
          // Set values to the residual of one obsveration
          e_mat.fill(e[j]);
          // Fill the output matrix (% represents element-wise multiplication, not modulo)
          XeeXh_i += ((k_mat % X.row(j).t()) * e_mat % (k_mat * ((d_row % distances.row(k)) % e.subvec(l, (l + n_obs_t - 1)).t()))) * X.rows(l, (l + n_obs_t - 1));
        }
        #pragma omp critical
        XeeXh += XeeXh_i;
      }
    } else {
      unsigned int j {};
      unsigned int k {};
      arma::mat e_mat(1, n_obs_t, arma::fill::zeros);
      // Loop over observations
      for(unsigned int i = 0; i < n_obs; i++) {
        j = i % n_obs_t;
        k = i - j;
        // Set values to the residual of one obsveration
        e_mat.fill(e[i]);
        // Fill the output matrix (% represents element-wise multiplication, not modulo)
        XeeXh += ((k_mat % X.row(i).t()) * e_mat % (k_mat * ((d_row % distances.row(j)) % e.subvec(k, (k + n_obs_t - 1)).t()))) * X.rows(k, (k + n_obs_t - 1));
      }
    }
  } else {
    // Cross-sectional case
    if(n_cores > 1) {
      // Loop over observations
      #pragma omp parallel for num_threads(n_cores)
      for(unsigned int i = 0; i < n_cores; i++) {
        arma::mat XeeXh_i(n_vars, n_vars, arma::fill::zeros);
        arma::mat e_mat(1, n_obs, arma::fill::zeros);
        for(unsigned int j = i; j < n_obs; j += n_cores) {
          // Set values to the residual of one obsveration
          e_mat.fill(e[j]);
          // Fill the output matrix (% represents element-wise multiplication, not modulo)
          XeeXh_i += ((k_mat % X.row(j).t()) * e_mat % (k_mat * ((d_row % distances.row(j)) % e.t()))) * X;
        }
        #pragma omp critical
        XeeXh += XeeXh_i;
      }
    } else {
      arma::mat e_mat(1, n_obs, arma::fill::zeros);
      // Loop over observations
      for(unsigned int i {0}; i < n_obs; i++) {
        e_mat.fill(e[i]);
        // Fill the output matrix (% represents element-wise multiplication, not modulo)
        XeeXh += ((k_mat % X.row(i).t()) * e_mat % (k_mat * ((d_row % distances.row(i)) % e.t()))) * X;
      }
    }
  }
  return XeeXh;
}

