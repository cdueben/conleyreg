#include <iostream>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#define ARMA_64BIT_WORD 1

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins(cpp11)]]

using namespace Rcpp;
using namespace arma;

// Function checking whether panel is balanced
// [[Rcpp::export]]
int isbalancedcpp(arma::mat M) {
  arma::vec ut = arma::unique(M.col(0));
  int utn = ut.n_elem;
  if((M.n_rows % utn) != 0) {
    return 0;
  } else {
    int upg = M.n_rows / utn;
    arma::mat fgu = M.submat(0, 1, upg - 1, 1);
    arma::vec fguu = arma::unique(fgu);
    if(fguu.n_elem < fgu.n_rows) {
      return 2;
    }
    for(size_t i {1}; i < (unsigned)utn ; ++i) {
      if(!arma::approx_equal(M.submat(i * upg, 1, (i + 1) * upg - 1, 1), fgu, "reldiff", 0.0001)) {
        return 0;
      }
    }
    return 1;
  }
}

// Function computing haversine distance
double haversine_dist(double lat1, double lat2, double lon1, double lon2) {

  double term1 = pow(sin((lat2 - lat1) / 2), 2);
  double term2 = cos(lat1) * cos(lat2) * pow(sin((lon2 - lon1)/2), 2);
  double the_terms = term1 + term2;
  double delta_sigma = 2 * atan2(sqrt(the_terms), sqrt(1-the_terms));

  return (6371.01 * delta_sigma);
}

// Function computing matrix of haversine distances
// [[Rcpp::export]]
arma::mat haversine_mat(arma::mat coords, unsigned long int n_obs) {
  arma::mat distances(n_obs, n_obs, fill::zeros);
  coords *= 0.01745329252;
  double dist {};

  for(unsigned long int i {0}; i < n_obs; i++) {
    for(unsigned long int j = i + 1; j < n_obs; j++) {
      dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
      distances(i, j) = dist;
      distances(j, i) = dist;
    }
  }

  return distances;
}

// Function calculating spatial sandwich
// [[Rcpp::export]]
arma::mat XeeXhC(arma::mat distances, arma::mat X, arma::vec e, unsigned long int n_obs_t, unsigned int n_vars) {

  // Set empty output matrix
  arma::mat XeeXh(n_vars, n_vars, fill::zeros);

  // Generate a [N variables x 1] matrix with all values set to one
  arma::mat k_mat(n_vars, 1, fill::ones);

  // Generate a [1 x obs in that period] matrix
  arma::mat d_row(1, n_obs_t, fill::ones);

  // Loop over observations
  for(unsigned long int i {0}; i < n_obs_t; i++) {
    // Generate a [1 x obs in that period] matrix with all values set to the residual of one obsveration
    arma::mat e_mat(1, n_obs_t, fill::zeros);
    e_mat.fill(e[i]);

    // Fill the output matrix (% represents element-wise multiplication, not modulo)
    XeeXh += ((k_mat % X.row(i).t()) * e_mat % (k_mat * ((d_row % distances.row(i)) % e.t()))) * X;
  }
  return XeeXh;
}

// Function calculating filling of spatial sandwich in logit and probit case
// [[Rcpp::export]]
arma::mat lp_filling(arma::mat distances, arma::mat X, arma::vec e, unsigned long int n_obs_t, unsigned int n_vars) {

  // Set empty output matrix
  arma::mat XeeXh(n_vars, n_vars, fill::zeros);

  arma::vec hf(n_obs_t);
  arma::mat cf(n_vars, n_vars, fill::zeros);
  arma::mat f(n_vars, n_vars, fill::zeros);

  // Loop over observations
  for(unsigned long int i {0}; i < n_obs_t; i++) {
    // Loop over variables
    for(unsigned int j {0}; j < n_vars; j++) {
      hf = X(i,j) * e * e(i) % distances.col(i);
      cf.row(j) = hf.t() * X;
    }
    f += cf;
  }
  return f;
}


// Function targeting serial correlation
// [[Rcpp::export]]
arma::mat time_dist(arma::vec times, double lag_cutoff, arma::mat X, arma::vec res, unsigned int n_obs_u, int n_vars) {

  // Generate a [obs of that unit x obs of that unit] matrix with all values set to one
  arma::mat dmat(n_obs_u, n_obs_u, fill::ones);

  for(unsigned int i {0}; i < n_obs_u; i++) {
    arma::vec t_diff {times};
    t_diff -= times[i];
    t_diff = abs(t_diff);

    // Generate vectors with lengths set to number obs of that unit
    arma::vec v1(n_obs_u);
    arma::vec v2(n_obs_u);

    for(unsigned int j {0}; j < n_obs_u; j++) {
      v1[j] = t_diff[j] <= lag_cutoff;
      v2[j] = t_diff[j] != t_diff[i];
      t_diff[j] = v1[j] * v2[j] * (1 - t_diff[j] / (lag_cutoff + 1));
    }

    dmat.row(i) %= t_diff.t();
  }

  // Generate a [N variables x N variables] matrix with all values set to zero
  arma::mat XeeXh(n_vars, n_vars, fill::zeros);

  // Generate a [N variables x 1] matrix with all values set to one
  arma::mat k_mat(n_vars, 1, fill::ones);

  // Generate a [1 x obs of that unit] matrix with all values set to one
  arma::mat d_row(1, n_obs_u, fill::ones);

  // Generate vector of transposed residuals
  arma::rowvec t_res {res.t()};

  for(unsigned int i {0}; i < n_obs_u; i++) {
    // Generate a [1 x obs of that unit] matrix with all values set to the residual of one obsveration
    arma::mat e_mat(1, n_obs_u, fill::zeros);
    e_mat.fill(res[i]);

    XeeXh += ((k_mat % X.row(i).t()) * e_mat % (k_mat * ((d_row % dmat.row(i)) % t_res))) * X;
  }

  return XeeXh;
}

// Function calculating sandwich in logit and probit regressions
// [[Rcpp::export]]
arma::mat lp_vcov(arma::mat V, arma::mat filling, unsigned int n_vars) {
  arma::mat inv_hessian(n_vars, n_vars);
  inv_hessian = arma::inv(-1 * arma::inv_sympd(V));
  return (inv_hessian * filling * inv_hessian);
}

