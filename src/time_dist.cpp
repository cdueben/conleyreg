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

// Function targeting serial correlation
// [[Rcpp::export]]
arma::mat time_dist(arma::vec &times, double lag_cutoff, arma::mat &X, arma::vec &res, unsigned int n_obs_u, unsigned int n_vars, unsigned int n_cores) {
  // Generate a [obs of that unit x obs of that unit] matrix with all values set to one
  arma::mat dmat(n_obs_u, n_obs_u, arma::fill::ones);
  #pragma omp parallel for num_threads(n_cores) if(n_cores > 1)
  for(unsigned int i = 0; i < n_obs_u; i++) {
    arma::vec t_diff(times);
    t_diff -= times[i];
    t_diff = abs(t_diff);
    // Generate vectors with lengths set to number obs of that unit
    arma::vec v1(n_obs_u);
    arma::vec v2(n_obs_u);
    for(unsigned int j = 0; j < n_obs_u; j++) {
      v1[j] = t_diff[j] <= lag_cutoff;
      v2[j] = t_diff[j] != t_diff[i];
      t_diff[j] = v1[j] * v2[j] * (1 - t_diff[j] / (lag_cutoff + 1));
    }
    dmat.row(i) %= t_diff.t();
  }
  // Generate a [N variables x N variables] matrix with all values set to zero
  arma::mat XeeXh(n_vars, n_vars, arma::fill::zeros);
  // Generate a [N variables x 1] matrix with all values set to one
  arma::mat k_mat(n_vars, 1, arma::fill::ones);
  // Generate a [1 x obs of that unit] matrix with all values set to one
  arma::mat d_row(1, n_obs_u, arma::fill::ones);
  // Generate vector of transposed residuals
  arma::rowvec t_res {res.t()};
  if(n_cores > 1) {
    #pragma omp parallel for num_threads(n_cores)
    for(unsigned int i = 0; i < n_cores; i++) {
      // Generate a [1 x obs of that unit] matrix with all values set to the residual of one obsveration
      arma::mat e_mat(1, n_obs_u, arma::fill::zeros);
      arma::mat XeeXh_i(n_vars, n_vars, arma::fill::zeros);
      for(unsigned int j = i; j < n_obs_u; j += n_cores) {
        e_mat.fill(res[j]);
        XeeXh_i += ((k_mat % X.row(j).t()) * e_mat % (k_mat * ((d_row % dmat.row(j)) % t_res))) * X;
      }
      #pragma omp critical
      XeeXh += XeeXh_i;
    }
  } else {
    // Generate a [1 x obs of that unit] matrix with all values set to the residual of one obsveration
    arma::mat e_mat(1, n_obs_u, arma::fill::zeros);
    for(unsigned int i {0}; i < n_obs_u; i++) {
      e_mat.fill(res[i]);
      XeeXh += ((k_mat % X.row(i).t()) * e_mat % (k_mat * ((d_row % dmat.row(i)) % t_res))) * X;
    }
  }
  return XeeXh;
}
