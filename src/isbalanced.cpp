#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

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
    unsigned int u_utn = (unsigned)utn;
    for(size_t i {1}; i < u_utn; ++i) {
      if(!arma::approx_equal(M.submat(i * upg, 1, (i + 1) * upg - 1, 1), fgu, "reldiff", 0.0001)) {
        return 0;
      }
    }
    return 1;
  }
}

