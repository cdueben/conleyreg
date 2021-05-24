#include <iostream>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins(cpp11)]]

using namespace Rcpp;
using namespace arma;

// Function overview
// 1 Function checking whether panel is balanced
// 2 Function computing haversine distance (called by functions 3)
// 3 Functions computing matrix of haversine distances (in panel application and logit and probit case)
// 3a Function computing dense matrix of haversine distances using a bartlett kernel
// 3b Function computing dense matrix of haversine distances using a uniform kernel
// 3c Function computing sparse matrix of haversine distances using a bartlett kernel
// 3d Function computing sparse matrix of haversine distances using a uniform kernel
// 3e Function computing sparse matrix of haversine distances using a bartlett kernel and batch insert
// 3f Function computing sparse matrix of haversine distances using a uniform kernel and batch insert
// 4 Functions calculating spatial sandwich in panel application
// 4a Function calculating spatial sandwich using a dense distance matrix computed with a bartlett kernel
// 4b Function calculating spatial sandwich using a dense distance matrix computed with a uniform kernel
// 4c Function calculating spatial sandwich using a sparse distance matrix (R does not export to arma::SpMat<short> objects in C++; thus bartlett and uniform use same type)
// 5 Functions computing matrix of haversine distances and calculating spatial sandwich in cross-sectional ols case (increases efficiency by combining functions 3 and 4,
// repetitive as important features that facilitate aggregating functions were added after C++11)
// 5a Function computing dense matrix of haversine distances with a bartlett kernel and calculating spatial sandwich in cross-sectional ols case
// 5b Function computing dense matrix of haversine distances with a uniform kernel and calculating spatial sandwich in cross-sectional ols case
// 5c Function computing sparse matrix of haversine distances with a bartlett kernel and calculating spatial sandwich in cross-sectional ols case
// 5d Function computing sparse matrix of haversine distances with a uniform kernel and calculating spatial sandwich in cross-sectional ols case
// 5e Function computing sparse matrix of haversine distances with a bartlett kernel and batch insert and calculating spatial sandwich in cross-sectional ols case
// 5f Function computing sparse matrix of haversine distances with a uniform kernel and batch insert and calculating spatial sandwich in cross-sectional ols case
// 6 Function calculating filling of spatial sandwich in logit and probit case
// 7 Functions calculating matrix of haversine distances and filling of spatial sandwich in logit and probit case
// 7a Function calculating dense matrix of haversine distances with a bartlett kernel and filling of spatial sandwich in logit and probit case
// 7b Function calculating dense matrix of haversine distances with a uniform kernel and filling of spatial sandwich in logit and probit case
// 7c Function calculating sparse matrix of haversine distances with a bartlett kernel and filling of spatial sandwich in logit and probit case
// 7d Function calculating sparse matrix of haversine distances with a uniform kernel and filling of spatial sandwich in logit and probit case
// 7e Function calculating sparse matrix of haversine distances with a bartlett kernel and batch insert and filling of spatial sandwich in logit and probit case
// 7f Function calculating sparse matrix of haversine distances with a uniform kernel and batch insert and filling of spatial sandwich in logit and probit case
// 8 Function targeting serial correlation
// 9 Function calculating sandwich in logit and probit regressions

// 1 Function checking whether panel is balanced
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

// 2 Function computing haversine distance
double haversine_dist(double lat1, double lat2, double lon1, double lon2) {
  double term1 = pow(sin((lat2 - lat1) / 2), 2);
  double term2 = cos(lat1) * cos(lat2) * pow(sin((lon2 - lon1)/2), 2);
  double the_terms = term1 + term2;
  double delta_sigma = 2 * atan2(sqrt(the_terms), sqrt(1-the_terms));
  return (6371.01 * delta_sigma);
}

// 3a Function computing dense matrix of haversine distances using a bartlett kernel
// [[Rcpp::export]]
arma::mat haversine_mat_b(arma::mat coords, unsigned int n_obs, double dist_cutoff) {
  arma::mat distances(n_obs, n_obs, fill::zeros);
  coords *= 0.01745329252;
  double dist {};
  for(unsigned int i {0}; i < n_obs; i++) {
    for(unsigned int j = i + 1; j < n_obs; j++) {
      dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
      if(dist < dist_cutoff) {
        dist = 1.0 - dist / dist_cutoff;
        distances.at(i, j) = dist;
        distances.at(j, i) = dist;
      }
    }
  }
  distances.diag().ones();
  return distances;
}

// 3b Function computing dense matrix of haversine distances using a uniform kernel
// [[Rcpp::export]]
arma::Mat<short> haversine_mat_u(arma::mat coords, unsigned int n_obs, double dist_cutoff) {
  arma::Mat<short> distances(n_obs, n_obs, fill::zeros);
  coords *= 0.01745329252;
  double dist {};
  for(unsigned int i {0}; i < n_obs; i++) {
    for(unsigned int j = i + 1; j < n_obs; j++) {
      dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
      if(dist < dist_cutoff) {
        distances.at(i, j) = 1;
        distances.at(j, i) = 1;
      }
    }
  }
  distances.diag().ones();
  return distances;
}

// 3c Function computing sparse matrix of haversine distances using a bartlett kernel
// [[Rcpp::export]]
arma::sp_mat haversine_spmat_b(arma::mat coords, unsigned int n_obs, double dist_cutoff) {
  arma::sp_mat distances(n_obs, n_obs);
  coords *= 0.01745329252;
  double dist {};
  for(unsigned int i {0}; i < n_obs; i++) {
    for(unsigned int j = i + 1; j < n_obs; j++) {
      dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
      if(dist < dist_cutoff) {
        dist = 1.0 - dist / dist_cutoff;
        distances.at(i, j) = dist;
        distances.at(j, i) = dist;
      }
    }
  }
  distances.diag().ones();
  return distances;
}

// 3d Function computing sparse matrix of haversine distances using a uniform kernel
// [[Rcpp::export]]
arma::SpMat<short> haversine_spmat_u(arma::mat coords, unsigned int n_obs, double dist_cutoff) {
  arma::SpMat<short> distances(n_obs, n_obs);
  coords *= 0.01745329252;
  double dist {};
  for(unsigned int i {0}; i < n_obs; i++) {
    for(unsigned int j = i + 1; j < n_obs; j++) {
      dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
      if(dist < dist_cutoff) {
        distances.at(i, j) = 1;
        distances.at(j, i) = 1;
      }
    }
  }
  distances.diag().ones();
  return distances;
}

// 3e Function computing sparse matrix of haversine distances using a bartlett kernel and batch insert
// [[Rcpp::export]]
arma::sp_mat haversine_spmat_b_bi(arma::mat coords, unsigned int n_obs, double dist_cutoff) {
  std::vector<double> dist_v;
  std::vector<unsigned int> dist_i;
  std::vector<unsigned int> dist_j;
  coords *= 0.01745329252;
  double dist {};
  for(unsigned long int i {0}; i < n_obs; i++) {
    for(unsigned long int j = i + 1; j < n_obs; j++) {
      dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
      if(dist < dist_cutoff) {
        dist = 1.0 - dist / dist_cutoff;
        dist_v.push_back(dist);
        dist_i.push_back(i);
        dist_j.push_back(j);
      }
    }
  }
  coords.reset();
  unsigned int mat_size = dist_i.size();
  arma::umat index_mat(2, mat_size * 2);
  arma::vec values_vec(mat_size * 2);
  unsigned int j {};
  for(unsigned int i {0}; i < mat_size; i++) {
    j = i * 2;
    index_mat.at(0, j) = dist_i[i];
    index_mat.at(1, j) = dist_j[i];
    index_mat.at(0, j + 1) = dist_j[i];
    index_mat.at(1, j + 1) = dist_i[i];
    values_vec.at(j) = dist_v[i];
    values_vec.at(j + 1) = dist_v[i];
  }
  std::vector<double>().swap(dist_v);
  std::vector<unsigned int>().swap(dist_i);
  std::vector<unsigned int>().swap(dist_j);
  arma::sp_mat distances(index_mat, values_vec, n_obs, n_obs);
  distances.diag().ones();
  return distances;
}

// 3f Function computing sparse matrix of haversine distances using a uniform kernel and batch insert
// [[Rcpp::export]]
arma::SpMat<short> haversine_spmat_u_bi(arma::mat coords, unsigned int n_obs, double dist_cutoff) {
  std::vector<short> dist_v;
  std::vector<unsigned int> dist_i;
  std::vector<unsigned int> dist_j;
  coords *= 0.01745329252;
  double dist {};
  for(unsigned long int i {0}; i < n_obs; i++) {
    for(unsigned long int j = i + 1; j < n_obs; j++) {
      dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
      if(dist < dist_cutoff) {
        dist_v.push_back(1);
        dist_i.push_back(i);
        dist_j.push_back(j);
      }
    }
  }
  coords.reset();
  unsigned int mat_size = dist_i.size();
  arma::umat index_mat(2, mat_size * 2);
  arma::Col<short> values_vec(mat_size * 2);
  unsigned int j {};
  for(unsigned int i {0}; i < mat_size; i++) {
    j = i * 2;
    index_mat.at(0, j) = dist_i[i];
    index_mat.at(1, j) = dist_j[i];
    index_mat.at(0, j + 1) = dist_j[i];
    index_mat.at(1, j + 1) = dist_i[i];
    values_vec.at(j) = dist_v[i];
    values_vec.at(j + 1) = dist_v[i];
  }
  std::vector<short>().swap(dist_v);
  std::vector<unsigned int>().swap(dist_i);
  std::vector<unsigned int>().swap(dist_j);
  arma::SpMat<short> distances(index_mat, values_vec, n_obs, n_obs);
  distances.diag().ones();
  return distances;
}

// 4a Function calculating spatial sandwich using a dense distance matrix computed with a bartlett kernel
// [[Rcpp::export]]
arma::mat XeeXhC_b(arma::mat distances, arma::mat X, arma::vec e, unsigned int n_obs_t, unsigned int n_vars) {
  // Set empty output matrix
  arma::mat XeeXh(n_vars, n_vars, fill::zeros);
  // Generate a [N variables x 1] matrix with all values set to one
  arma::mat k_mat(n_vars, 1, fill::ones);
  // Generate a [1 x obs in that period] matrix
  arma::mat d_row(1, n_obs_t, fill::ones);
  // Loop over observations
  for(unsigned int i {0}; i < n_obs_t; i++) {
    // Generate a [1 x obs in that period] matrix with all values set to the residual of one obsveration
    arma::mat e_mat(1, n_obs_t, fill::zeros);
    e_mat.fill(e[i]);
    // Fill the output matrix (% represents element-wise multiplication, not modulo)
    XeeXh += ((k_mat % X.row(i).t()) * e_mat % (k_mat * ((d_row % distances.row(i)) % e.t()))) * X;
  }
  return XeeXh;
}

// 4b Function calculating spatial sandwich using a dense distance matrix computed with a uniform kernel
// [[Rcpp::export]]
arma::mat XeeXhC_u(arma::Mat<short> distances, arma::mat X, arma::vec e, unsigned int n_obs_t, unsigned int n_vars) {
  // Set empty output matrix
  arma::mat XeeXh(n_vars, n_vars, fill::zeros);
  // Generate a [N variables x 1] matrix with all values set to one
  arma::mat k_mat(n_vars, 1, fill::ones);
  // Generate a [1 x obs in that period] matrix
  arma::mat d_row(1, n_obs_t, fill::ones);
  // Loop over observations
  for(unsigned int i {0}; i < n_obs_t; i++) {
    // Generate a [1 x obs in that period] matrix with all values set to the residual of one obsveration
    arma::mat e_mat(1, n_obs_t, fill::zeros);
    e_mat.fill(e[i]);
    // Fill the output matrix (% represents element-wise multiplication, not modulo)
    XeeXh += ((k_mat % X.row(i).t()) * e_mat % (k_mat * ((d_row % distances.row(i)) % e.t()))) * X;
  }
  return XeeXh;
}

// 4c Function calculating spatial sandwich using a sparse distance matrix computed with a bartlett kernel
// [[Rcpp::export]]
arma::mat XeeXhCsp_b(arma::sp_mat distances, arma::mat X, arma::vec e, unsigned int n_obs_t, unsigned int n_vars) {
  // Set empty output matrix
  arma::mat XeeXh(n_vars, n_vars, fill::zeros);
  // Generate a [N variables x 1] matrix with all values set to one
  arma::mat k_mat(n_vars, 1, fill::ones);
  // Generate a [1 x obs in that period] matrix
  arma::mat d_row(1, n_obs_t, fill::ones);
  // Loop over observations
  for(unsigned int i {0}; i < n_obs_t; i++) {
    // Generate a [1 x obs in that period] matrix with all values set to the residual of one obsveration
    arma::mat e_mat(1, n_obs_t, fill::zeros);
    e_mat.fill(e[i]);
    // Fill the output matrix (% represents element-wise multiplication, not modulo)
    XeeXh += ((k_mat % X.row(i).t()) * e_mat % (k_mat * ((d_row % distances.row(i)) % e.t()))) * X;
  }
  return XeeXh;
}

// 5a Function computing dense matrix of haversine distances with a bartlett kernel and calculating spatial sandwich in cross-sectional ols case
// [[Rcpp::export]]
arma::mat haversine_mat_XeeXhC_b(arma::mat coords, unsigned int n_obs, double dist_cutoff, arma::mat X, arma::vec e, unsigned int n_vars) {
  arma::mat distances(n_obs, n_obs, fill::zeros);
  coords *= 0.01745329252;
  double dist {};
  for(unsigned int i {0}; i < n_obs; i++) {
    for(unsigned int j = i + 1; j < n_obs; j++) {
      dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
      if(dist < dist_cutoff) {
        dist = 1.0 - dist / dist_cutoff;
        distances.at(i, j) = dist;
        distances.at(j, i) = dist;
      }
    }
  }
  coords.reset();
  distances.diag().ones();
  // Set empty output matrix
  arma::mat XeeXh(n_vars, n_vars, fill::zeros);
  // Generate a [N variables x 1] matrix with all values set to one
  arma::mat k_mat(n_vars, 1, fill::ones);
  // Generate a [1 x obs in that period] matrix
  arma::mat d_row(1, n_obs, fill::ones);
  // Loop over observations
  for(unsigned int i {0}; i < n_obs; i++) {
    // Generate a [1 x obs in that period] matrix with all values set to the residual of one obsveration
    arma::mat e_mat(1, n_obs, fill::zeros);
    e_mat.fill(e[i]);
    // Fill the output matrix (% represents element-wise multiplication, not modulo)
    XeeXh += ((k_mat % X.row(i).t()) * e_mat % (k_mat * ((d_row % distances.row(i)) % e.t()))) * X;
  }
  return XeeXh;
}

// 5b Function computing dense matrix of haversine distances with a uniform kernel and calculating spatial sandwich in cross-sectional ols case
// [[Rcpp::export]]
arma::mat haversine_mat_XeeXhC_u(arma::mat coords, unsigned int n_obs, double dist_cutoff, arma::mat X, arma::vec e, unsigned int n_vars) {
  arma::Mat<short> distances(n_obs, n_obs, fill::zeros);
  coords *= 0.01745329252;
  double dist {};
  for(unsigned int i {0}; i < n_obs; i++) {
    for(unsigned int j = i + 1; j < n_obs; j++) {
      dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
      if(dist < dist_cutoff) {
        distances.at(i, j) = 1;
        distances.at(j, i) = 1;
      }
    }
  }
  coords.reset();
  distances.diag().ones();
  // Set empty output matrix
  arma::mat XeeXh(n_vars, n_vars, fill::zeros);
  // Generate a [N variables x 1] matrix with all values set to one
  arma::mat k_mat(n_vars, 1, fill::ones);
  // Generate a [1 x obs in that period] matrix
  arma::mat d_row(1, n_obs, fill::ones);
  // Loop over observations
  for(unsigned int i {0}; i < n_obs; i++) {
    // Generate a [1 x obs in that period] matrix with all values set to the residual of one obsveration
    arma::mat e_mat(1, n_obs, fill::zeros);
    e_mat.fill(e[i]);
    // Fill the output matrix (% represents element-wise multiplication, not modulo)
    XeeXh += ((k_mat % X.row(i).t()) * e_mat % (k_mat * ((d_row % distances.row(i)) % e.t()))) * X;
  }
  return XeeXh;
}

// 5c Function computing sparse matrix of haversine distances with a bartlett kernel and calculating spatial sandwich in cross-sectional ols case
// [[Rcpp::export]]
arma::mat haversine_spmat_XeeXhC_b(arma::mat coords, unsigned int n_obs, double dist_cutoff, arma::mat X, arma::vec e, unsigned int n_vars) {
  arma::sp_mat distances(n_obs, n_obs);
  coords *= 0.01745329252;
  double dist {};
  for(unsigned int i {0}; i < n_obs; i++) {
    for(unsigned int j = i + 1; j < n_obs; j++) {
      dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
      if(dist < dist_cutoff) {
        dist = 1.0 - dist / dist_cutoff;
        distances.at(i, j) = dist;
        distances.at(j, i) = dist;
      }
    }
  }
  coords.reset();
  distances.diag().ones();
  // Set empty output matrix
  arma::mat XeeXh(n_vars, n_vars, fill::zeros);
  // Generate a [N variables x 1] matrix with all values set to one
  arma::mat k_mat(n_vars, 1, fill::ones);
  // Generate a [1 x obs in that period] matrix
  arma::mat d_row(1, n_obs, fill::ones);
  // Loop over observations
  for(unsigned int i {0}; i < n_obs; i++) {
    // Generate a [1 x obs in that period] matrix with all values set to the residual of one obsveration
    arma::mat e_mat(1, n_obs, fill::zeros);
    e_mat.fill(e[i]);
    // Fill the output matrix (% represents element-wise multiplication, not modulo)
    XeeXh += ((k_mat % X.row(i).t()) * e_mat % (k_mat * ((d_row % distances.row(i)) % e.t()))) * X;
  }
  return XeeXh;
}

// 5d Function computing sparse matrix of haversine distances with a uniform kernel and calculating spatial sandwich in cross-sectional ols case
// [[Rcpp::export]]
arma::mat haversine_spmat_XeeXhC_u(arma::mat coords, unsigned int n_obs, double dist_cutoff, arma::mat X, arma::vec e, unsigned int n_vars) {
  arma::SpMat<short> distances(n_obs, n_obs);
  coords *= 0.01745329252;
  double dist {};
  for(unsigned int i {0}; i < n_obs; i++) {
    for(unsigned int j = i + 1; j < n_obs; j++) {
      dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
      if(dist < dist_cutoff) {
        distances.at(i, j) = 1;
        distances.at(j, i) = 1;
      }
    }
  }
  coords.reset();
  distances.diag().ones();
  // Set empty output matrix
  arma::mat XeeXh(n_vars, n_vars, fill::zeros);
  // Generate a [N variables x 1] matrix with all values set to one
  arma::mat k_mat(n_vars, 1, fill::ones);
  // Generate a [1 x obs in that period] matrix
  arma::mat d_row(1, n_obs, fill::ones);
  // Loop over observations
  for(unsigned int i {0}; i < n_obs; i++) {
    // Generate a [1 x obs in that period] matrix with all values set to the residual of one obsveration
    arma::mat e_mat(1, n_obs, fill::zeros);
    e_mat.fill(e[i]);
    // Fill the output matrix (% represents element-wise multiplication, not modulo)
    XeeXh += ((k_mat % X.row(i).t()) * e_mat % (k_mat * ((d_row % distances.row(i)) % e.t()))) * X;
  }
  return XeeXh;
}

// 5e Function computing sparse matrix of haversine distances with a bartlett kernel and batch insert and calculating spatial sandwich in cross-sectional ols case
// [[Rcpp::export]]
arma::mat haversine_spmat_XeeXhC_b_bi(arma::mat coords, unsigned int n_obs, double dist_cutoff, arma::mat X, arma::vec e, unsigned int n_vars) {
  std::vector<double> dist_v;
  std::vector<unsigned int> dist_i;
  std::vector<unsigned int> dist_j;
  coords *= 0.01745329252;
  double dist {};
  for(unsigned long int i {0}; i < n_obs; i++) {
    for(unsigned long int j = i + 1; j < n_obs; j++) {
      dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
      if(dist < dist_cutoff) {
        dist = 1.0 - dist / dist_cutoff;
        dist_v.push_back(dist);
        dist_i.push_back(i);
        dist_j.push_back(j);
      }
    }
  }
  coords.reset();
  unsigned int mat_size = dist_i.size();
  arma::umat index_mat(2, mat_size * 2);
  arma::vec values_vec(mat_size * 2);
  unsigned int j {};
  for(unsigned int i {0}; i < mat_size; i++) {
    j = i * 2;
    index_mat.at(0, j) = dist_i[i];
    index_mat.at(1, j) = dist_j[i];
    index_mat.at(0, j + 1) = dist_j[i];
    index_mat.at(1, j + 1) = dist_i[i];
    values_vec.at(j) = dist_v[i];
    values_vec.at(j + 1) = dist_v[i];
  }
  std::vector<double>().swap(dist_v);
  std::vector<unsigned int>().swap(dist_i);
  std::vector<unsigned int>().swap(dist_j);
  arma::sp_mat distances(index_mat, values_vec, n_obs, n_obs);
  distances.diag().ones();
  // Set empty output matrix
  arma::mat XeeXh(n_vars, n_vars, fill::zeros);
  // Generate a [N variables x 1] matrix with all values set to one
  arma::mat k_mat(n_vars, 1, fill::ones);
  // Generate a [1 x obs in that period] matrix
  arma::mat d_row(1, n_obs, fill::ones);
  // Loop over observations
  for(unsigned int i {0}; i < n_obs; i++) {
    // Generate a [1 x obs in that period] matrix with all values set to the residual of one obsveration
    arma::mat e_mat(1, n_obs, fill::zeros);
    e_mat.fill(e[i]);
    // Fill the output matrix (% represents element-wise multiplication, not modulo)
    XeeXh += ((k_mat % X.row(i).t()) * e_mat % (k_mat * ((d_row % distances.row(i)) % e.t()))) * X;
  }
  return XeeXh;
}

// 5f Function computing sparse matrix of haversine distances with a uniform kernel and batch insert and calculating spatial sandwich in cross-sectional ols case
// [[Rcpp::export]]
arma::mat haversine_spmat_XeeXhC_u_bi(arma::mat coords, unsigned int n_obs, double dist_cutoff, arma::mat X, arma::vec e, unsigned int n_vars) {
  std::vector<short> dist_v;
  std::vector<unsigned int> dist_i;
  std::vector<unsigned int> dist_j;
  coords *= 0.01745329252;
  double dist {};
  for(unsigned long int i {0}; i < n_obs; i++) {
    for(unsigned long int j = i + 1; j < n_obs; j++) {
      dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
      if(dist < dist_cutoff) {
        dist_v.push_back(1);
        dist_i.push_back(i);
        dist_j.push_back(j);
      }
    }
  }
  coords.reset();
  unsigned int mat_size = dist_i.size();
  arma::umat index_mat(2, mat_size * 2);
  arma::Col<short> values_vec(mat_size * 2);
  unsigned int j {};
  for(unsigned int i {0}; i < mat_size; i++) {
    j = i * 2;
    index_mat.at(0, j) = dist_i[i];
    index_mat.at(1, j) = dist_j[i];
    index_mat.at(0, j + 1) = dist_j[i];
    index_mat.at(1, j + 1) = dist_i[i];
    values_vec.at(j) = dist_v[i];
    values_vec.at(j + 1) = dist_v[i];
  }
  std::vector<short>().swap(dist_v);
  std::vector<unsigned int>().swap(dist_i);
  std::vector<unsigned int>().swap(dist_j);
  arma::SpMat<short> distances(index_mat, values_vec, n_obs, n_obs);
  distances.diag().ones();
  // Set empty output matrix
  arma::mat XeeXh(n_vars, n_vars, fill::zeros);
  // Generate a [N variables x 1] matrix with all values set to one
  arma::mat k_mat(n_vars, 1, fill::ones);
  // Generate a [1 x obs in that period] matrix
  arma::mat d_row(1, n_obs, fill::ones);
  // Loop over observations
  for(unsigned int i {0}; i < n_obs; i++) {
    // Generate a [1 x obs in that period] matrix with all values set to the residual of one obsveration
    arma::mat e_mat(1, n_obs, fill::zeros);
    e_mat.fill(e[i]);
    // Fill the output matrix (% represents element-wise multiplication, not modulo)
    XeeXh += ((k_mat % X.row(i).t()) * e_mat % (k_mat * ((d_row % distances.row(i)) % e.t()))) * X;
  }
  return XeeXh;
}

// 6 Function calculating filling of spatial sandwich in logit and probit case
// [[Rcpp::export]]
arma::mat lp_filling(arma::mat distances, arma::mat X, arma::vec e, unsigned int n_obs_t, unsigned int n_vars) {
  // Set empty output matrix
  arma::mat XeeXh(n_vars, n_vars, fill::zeros);
  arma::vec hf(n_obs_t);
  arma::mat cf(n_vars, n_vars, fill::zeros);
  arma::mat f(n_vars, n_vars, fill::zeros);
  // Loop over observations
  for(unsigned int i {0}; i < n_obs_t; i++) {
    // Loop over variables
    for(unsigned int j {0}; j < n_vars; j++) {
      hf = X(i,j) * e * e(i) % distances.col(i);
      cf.row(j) = hf.t() * X;
    }
    f += cf;
  }
  return f;
}

// 7a Function calculating dense matrix of haversine distances with a bartlett kernel and filling of spatial sandwich in logit and probit case
// [[Rcpp::export]]
arma::mat haversine_mat_lp_b(arma::mat coords, arma::mat X, arma::vec e, unsigned int n_obs, unsigned int n_vars, double dist_cutoff) {
  arma::mat distances(n_obs, n_obs, fill::zeros);
  coords *= 0.01745329252;
  double dist {};
  for(unsigned int i {0}; i < n_obs; i++) {
    for(unsigned int j = i + 1; j < n_obs; j++) {
      dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
      if(dist < dist_cutoff) {
        dist = 1.0 - dist / dist_cutoff;
        distances.at(i, j) = dist;
        distances.at(j, i) = dist;
      }
    }
  }
  coords.reset();
  distances.diag().ones();
  // Set empty output matrix
  arma::mat XeeXh(n_vars, n_vars, fill::zeros);
  arma::vec hf(n_obs);
  arma::mat cf(n_vars, n_vars, fill::zeros);
  arma::mat f(n_vars, n_vars, fill::zeros);
  // Loop over observations
  for(unsigned int i {0}; i < n_obs; i++) {
    // Loop over variables
    for(unsigned int j {0}; j < n_vars; j++) {
      hf = X(i,j) * e * e(i) % distances.col(i);
      cf.row(j) = hf.t() * X;
    }
    f += cf;
  }
  return f;
}

// 7b Function calculating dense matrix of haversine distances with a uniform kernel and filling of spatial sandwich in logit and probit case
// [[Rcpp::export]]
arma::mat haversine_mat_lp_u(arma::mat coords, arma::mat X, arma::vec e, unsigned int n_obs, unsigned int n_vars, double dist_cutoff) {
  arma::Mat<short> distances(n_obs, n_obs, fill::zeros);
  coords *= 0.01745329252;
  double dist {};
  for(unsigned int i {0}; i < n_obs; i++) {
    for(unsigned int j = i + 1; j < n_obs; j++) {
      dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
      if(dist < dist_cutoff) {
        distances.at(i, j) = 1;
        distances.at(j, i) = 1;
      }
    }
  }
  coords.reset();
  distances.diag().ones();
  // Set empty output matrix
  arma::mat XeeXh(n_vars, n_vars, fill::zeros);
  arma::vec hf(n_obs);
  arma::mat cf(n_vars, n_vars, fill::zeros);
  arma::mat f(n_vars, n_vars, fill::zeros);
  // Loop over observations
  for(unsigned int i {0}; i < n_obs; i++) {
    // Loop over variables
    for(unsigned int j {0}; j < n_vars; j++) {
      hf = X(i,j) * e * e(i) % distances.col(i);
      cf.row(j) = hf.t() * X;
    }
    f += cf;
  }
  return f;
}

// 7c Function calculating sparse matrix of haversine distances with a bartlett kernel and filling of spatial sandwich in logit and probit case
// [[Rcpp::export]]
arma::mat haversine_spmat_lp_b(arma::mat coords, arma::mat X, arma::vec e, unsigned int n_obs, unsigned int n_vars, double dist_cutoff) {
  arma::sp_mat distances(n_obs, n_obs);
  coords *= 0.01745329252;
  double dist {};
  for(unsigned int i {0}; i < n_obs; i++) {
    for(unsigned int j = i + 1; j < n_obs; j++) {
      dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
      if(dist < dist_cutoff) {
        dist = 1.0 - dist / dist_cutoff;
        distances.at(i, j) = dist;
        distances.at(j, i) = dist;
      }
    }
  }
  coords.reset();
  distances.diag().ones();
  // Set empty output matrix
  arma::mat XeeXh(n_vars, n_vars, fill::zeros);
  arma::vec hf(n_obs);
  arma::mat cf(n_vars, n_vars, fill::zeros);
  arma::mat f(n_vars, n_vars, fill::zeros);
  // Loop over observations
  for(unsigned int i {0}; i < n_obs; i++) {
    // Loop over variables
    for(unsigned int j {0}; j < n_vars; j++) {
      hf = X(i,j) * e * e(i) % distances.col(i);
      cf.row(j) = hf.t() * X;
    }
    f += cf;
  }
  return f;
}

// 7d Function calculating sparse matrix of haversine distances with a uniform kernel and filling of spatial sandwich in logit and probit case
// [[Rcpp::export]]
arma::mat haversine_spmat_lp_u(arma::mat coords, arma::mat X, arma::vec e, unsigned int n_obs, unsigned int n_vars, double dist_cutoff) {
  arma::SpMat<short> distances(n_obs, n_obs);
  coords *= 0.01745329252;
  double dist {};
  for(unsigned int i {0}; i < n_obs; i++) {
    for(unsigned int j = i + 1; j < n_obs; j++) {
      dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
      if(dist < dist_cutoff) {
        distances.at(i, j) = 1;
        distances.at(j, i) = 1;
      }
    }
  }
  coords.reset();
  distances.diag().ones();
  // Set empty output matrix
  arma::mat XeeXh(n_vars, n_vars, fill::zeros);
  arma::vec hf(n_obs);
  arma::mat cf(n_vars, n_vars, fill::zeros);
  arma::mat f(n_vars, n_vars, fill::zeros);
  // Loop over observations
  for(unsigned int i {0}; i < n_obs; i++) {
    // Loop over variables
    for(unsigned int j {0}; j < n_vars; j++) {
      hf = X(i,j) * e * e(i) % distances.col(i);
      cf.row(j) = hf.t() * X;
    }
    f += cf;
  }
  return f;
}

// 7e Function calculating sparse matrix of haversine distances with a bartlett kernel and batch insert and filling of spatial sandwich in logit and probit case
// [[Rcpp::export]]
arma::mat haversine_spmat_lp_b_bi(arma::mat coords, arma::mat X, arma::vec e, unsigned int n_obs, unsigned int n_vars, double dist_cutoff) {
  std::vector<double> dist_v;
  std::vector<unsigned int> dist_i;
  std::vector<unsigned int> dist_j;
  coords *= 0.01745329252;
  double dist {};
  for(unsigned long int i {0}; i < n_obs; i++) {
    for(unsigned long int j = i + 1; j < n_obs; j++) {
      dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
      if(dist < dist_cutoff) {
        dist = 1.0 - dist / dist_cutoff;
        dist_v.push_back(dist);
        dist_i.push_back(i);
        dist_j.push_back(j);
      }
    }
  }
  coords.reset();
  unsigned int mat_size = dist_i.size();
  arma::umat index_mat(2, mat_size * 2);
  arma::vec values_vec(mat_size * 2);
  unsigned int j {};
  for(unsigned int i {0}; i < mat_size; i++) {
    j = i * 2;
    index_mat.at(0, j) = dist_i[i];
    index_mat.at(1, j) = dist_j[i];
    index_mat.at(0, j + 1) = dist_j[i];
    index_mat.at(1, j + 1) = dist_i[i];
    values_vec.at(j) = dist_v[i];
    values_vec.at(j + 1) = dist_v[i];
  }
  std::vector<double>().swap(dist_v);
  std::vector<unsigned int>().swap(dist_i);
  std::vector<unsigned int>().swap(dist_j);
  arma::sp_mat distances(index_mat, values_vec, n_obs, n_obs);
  distances.diag().ones();
  // Set empty output matrix
  arma::mat XeeXh(n_vars, n_vars, fill::zeros);
  arma::vec hf(n_obs);
  arma::mat cf(n_vars, n_vars, fill::zeros);
  arma::mat f(n_vars, n_vars, fill::zeros);
  // Loop over observations
  for(unsigned int i {0}; i < n_obs; i++) {
    // Loop over variables
    for(unsigned int j {0}; j < n_vars; j++) {
      hf = X(i,j) * e * e(i) % distances.col(i);
      cf.row(j) = hf.t() * X;
    }
    f += cf;
  }
  return f;
}

// 7f Function calculating sparse matrix of haversine distances with a uniform kernel and batch insert and filling of spatial sandwich in logit and probit case
// [[Rcpp::export]]
arma::mat haversine_spmat_lp_u_bi(arma::mat coords, arma::mat X, arma::vec e, unsigned int n_obs, unsigned int n_vars, double dist_cutoff) {
  std::vector<short> dist_v;
  std::vector<unsigned int> dist_i;
  std::vector<unsigned int> dist_j;
  coords *= 0.01745329252;
  double dist {};
  for(unsigned long int i {0}; i < n_obs; i++) {
    for(unsigned long int j = i + 1; j < n_obs; j++) {
      dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
      if(dist < dist_cutoff) {
        dist_v.push_back(1);
        dist_i.push_back(i);
        dist_j.push_back(j);
      }
    }
  }
  coords.reset();
  unsigned int mat_size = dist_i.size();
  arma::umat index_mat(2, mat_size * 2);
  arma::Col<short> values_vec(mat_size * 2);
  unsigned int j {};
  for(unsigned int i {0}; i < mat_size; i++) {
    j = i * 2;
    index_mat.at(0, j) = dist_i[i];
    index_mat.at(1, j) = dist_j[i];
    index_mat.at(0, j + 1) = dist_j[i];
    index_mat.at(1, j + 1) = dist_i[i];
    values_vec.at(j) = dist_v[i];
    values_vec.at(j + 1) = dist_v[i];
  }
  std::vector<short>().swap(dist_v);
  std::vector<unsigned int>().swap(dist_i);
  std::vector<unsigned int>().swap(dist_j);
  arma::SpMat<short> distances(index_mat, values_vec, n_obs, n_obs);
  distances.diag().ones();
  // Set empty output matrix
  arma::mat XeeXh(n_vars, n_vars, fill::zeros);
  arma::vec hf(n_obs);
  arma::mat cf(n_vars, n_vars, fill::zeros);
  arma::mat f(n_vars, n_vars, fill::zeros);
  // Loop over observations
  for(unsigned int i {0}; i < n_obs; i++) {
    // Loop over variables
    for(unsigned int j {0}; j < n_vars; j++) {
      hf = X(i,j) * e * e(i) % distances.col(i);
      cf.row(j) = hf.t() * X;
    }
    f += cf;
  }
  return f;
}

// 8 Function targeting serial correlation
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

// 9 Function calculating sandwich in logit and probit regressions
// [[Rcpp::export]]
arma::mat lp_vcov(arma::mat V, arma::mat filling, unsigned int n_vars) {
  arma::mat inv_hessian(n_vars, n_vars);
  inv_hessian = arma::inv(-1 * arma::inv_sympd(V));
  return (inv_hessian * filling * inv_hessian);
}

