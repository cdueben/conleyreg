#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
#include <cmath>

// Function calculating sandwich in logit and probit regressions
// [[Rcpp::export]]
arma::mat lp_vcov(arma::mat &V, arma::mat &filling, unsigned int n_vars) {
  arma::mat inv_hessian(n_vars, n_vars);
  inv_hessian = arma::inv(-1 * arma::inv_sympd(V));
  return (inv_hessian * filling * inv_hessian);
}
