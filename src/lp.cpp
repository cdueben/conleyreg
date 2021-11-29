#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
#include <omp.h>
// [[Rcpp::plugins(openmp)]]
#include <cmath>
#include <vector>
#include <cstddef>
#include "distance_functions.h"
#include "distance_matrices.h"
#include "lp_filling.h"

// Function overview
// 1 lp Function calculating distance matrix and filling of spatial sandwich in logit and probit case
// 2 lp_d_b Function calculating sparse distance matrix with a bartlett kernel and batch insert and filling of spatial sandwich in logit and probit case
// 3 lp_f_b Function calculating sparse distance matrix with a bartlett kernel and batch insert using floats and filling of spatial sandwich in logit and probit case
// 4 lp_d_b_p Function parallely calculating sparse distance matrix with a bartlett kernel and batch insert and filling of spatial sandwich in logit and probit case
// 5 lp_f_b_p Function parallely calculating sparse distance matrix with a bartlett kernel and batch insert using floats and filling of spatial sandwich in logit and probit case
// 6 lp_s_b Function calculating sparse distance matrix with a uniform kernel and batch insert and filling of spatial sandwich in logit and probit case
// 7 lp_s_b_p Function parallely calculating sparse distance matrix with a uniform kernel and batch insert and filling of spatial sandwich in logit and probit case

// 1 Function calculating distance matrix and filling of spatial sandwich in logit and probit case
// [[Rcpp::export]]
arma::mat lp(arma::mat &coords, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_vars, double dist_cutoff, bool haversine, bool sparse, bool bartlett,
  bool flt, unsigned int n_cores) {
  arma::mat distances(n_obs, n_obs, arma::fill::zeros);
  if(sparse) {
    if(bartlett) {
      if(flt) {
        // Bartlett, sparse, float
        arma::sp_fmat distances(n_obs, n_obs);
        dist_spmat_f(distances, coords, n_obs, dist_cutoff, haversine, n_cores);
        arma::mat f = lp_filling_s_f(distances, X, e, n_obs, n_vars, n_cores);
        return f;
      } else {
        // Bartlett, sparse, double
        arma::sp_mat distances(n_obs, n_obs);
        dist_spmat_d(distances, coords, n_obs, dist_cutoff, haversine, n_cores);
        arma::mat f = lp_filling_s_d(distances, X, e, n_obs, n_vars, n_cores);
        return f;
      }
    } else {
      // Uniform, sparse
      arma::SpMat<short> distances(n_obs, n_obs);
      dist_spmat_s(distances, coords, n_obs, dist_cutoff, haversine, n_cores);
      arma::mat f = lp_filling_s_s(distances, X, e, n_obs, n_vars, n_cores);
      return f;
    }
  } else {
    if(bartlett) {
      if(flt) {
        // Bartlett, dense, float
        arma::fmat distances(n_obs, n_obs, arma::fill::zeros);
        dist_mat_f(distances, coords, n_obs, dist_cutoff, haversine, n_cores);
        arma::mat f = lp_filling_d_f(distances, X, e, n_obs, n_vars, n_cores);
        return f;
      } else {
        // Bartlett, dense, double
        arma::mat distances(n_obs, n_obs, arma::fill::zeros);
        dist_mat_d(distances, coords, n_obs, dist_cutoff, haversine, n_cores);
        arma::mat f = lp_filling_d_d(distances, X, e, n_obs, n_vars, n_cores);
        return f;
      }
    } else {
      // Uniform, dense
      arma::Mat<short> distances(n_obs, n_obs, arma::fill::zeros);
      dist_mat_s(distances, coords, n_obs, dist_cutoff, haversine, n_cores);
      arma::mat f = lp_filling_d_s(distances, X, e, n_obs, n_vars, n_cores);
      return f;
    }
  }
}

// 2 Function calculating sparse distance matrix with a bartlett kernel and batch insert and filling of spatial sandwich in logit and probit case
// [[Rcpp::export]]
arma::mat lp_d_b(arma::mat &coords, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_vars, double dist_cutoff, bool haversine,
  unsigned short int batch_ram_opt) {
  std::vector<double> dist_v;
  std::vector<std::vector<unsigned int>> dist_j(n_obs);
  double dist {};
  if(haversine) {
    for(unsigned long int i {0}; i < n_obs; i++) {
      for(unsigned long int j = i + 1; j < n_obs; j++) {
        dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          dist = 1.0 - dist / dist_cutoff;
          dist_v.push_back(dist);
          dist_j[i].push_back(j);
        }
      }
    }
  } else {
    for(unsigned long int i {0}; i < n_obs; i++) {
      for(unsigned long int j = i + 1; j < n_obs; j++) {
        dist = euclidean_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          dist = 1.0 - dist / dist_cutoff;
          dist_v.push_back(dist);
          dist_j[i].push_back(j);
        }
      }
    }
  }
  arma::umat index_mat(2, dist_v.size());
  arma::vec values_vec(dist_v);
  if(batch_ram_opt > 1) {
    std::vector<double>().swap(dist_v);
  }
  std::size_t k {0};
  unsigned int vec_length_i {};
  for(unsigned int i {0}; i < n_obs; i++) {
    vec_length_i = dist_j[i].size();
    for(std::size_t j {0}; j < vec_length_i; j++) {
      index_mat.at(0, k) = i;
      index_mat.at(1, k) = dist_j[i][j];
      k++;
    }
  }
  if(batch_ram_opt > 1) {
    std::vector<std::vector<unsigned int>>().swap(dist_j);
  }
  arma::sp_mat distances(index_mat, values_vec, n_obs, n_obs);
  distances.diag().ones();
  distances = arma::symmatu(distances);
  arma::mat f = lp_filling_s_d(distances, X, e, n_obs, n_vars, 1);
  return f;
}

// 3 Function calculating sparse distance matrix with a bartlett kernel and batch insert using floats and filling of spatial sandwich in logit and probit case
// [[Rcpp::export]]
arma::mat lp_f_b(arma::mat &coords, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_vars, double dist_cutoff, bool haversine,
  unsigned short int batch_ram_opt) {
  std::vector<float> dist_v;
  std::vector<std::vector<unsigned int>> dist_j(n_obs);
  double dist {};
  float dist_f {};
  if(haversine) {
    for(unsigned long int i {0}; i < n_obs; i++) {
      for(unsigned long int j = i + 1; j < n_obs; j++) {
        dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          dist_f = (float) (1.0 - dist / dist_cutoff);
          dist_v.push_back(dist_f);
          dist_j[i].push_back(j);
        }
      }
    }
  } else {
    for(unsigned long int i {0}; i < n_obs; i++) {
      for(unsigned long int j = i + 1; j < n_obs; j++) {
        dist = euclidean_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          dist_f = (float) (1.0 - dist / dist_cutoff);
          dist_v.push_back(dist_f);
          dist_j[i].push_back(j);
        }
      }
    }
  }
  arma::umat index_mat(2, dist_v.size());
  arma::fvec values_vec(dist_v);
  if(batch_ram_opt > 1) {
    std::vector<float>().swap(dist_v);
  }
  std::size_t k {0};
  unsigned int vec_length_i {};
  for(unsigned int i {0}; i < n_obs; i++) {
    vec_length_i = dist_j[i].size();
    for(std::size_t j {0}; j < vec_length_i; j++) {
      index_mat.at(0, k) = i;
      index_mat.at(1, k) = dist_j[i][j];
      k++;
    }
  }
  if(batch_ram_opt > 1) {
    std::vector<std::vector<unsigned int>>().swap(dist_j);
  }
  arma::sp_fmat distances(index_mat, values_vec, n_obs, n_obs);
  distances.diag().ones();
  distances = arma::symmatu(distances);
  arma::mat f = lp_filling_s_f(distances, X, e, n_obs, n_vars, 1);
  return f;
}

// 4 Function parallely calculating sparse distance matrix with a bartlett kernel and batch insert and filling of spatial sandwich in logit and probit case
// [[Rcpp::export]]
arma::mat lp_d_b_p(arma::mat &coords, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_vars, double dist_cutoff, bool haversine,
  unsigned short int batch_ram_opt, unsigned int n_cores) {
  std::vector<std::vector<double>> dist_v(n_obs);
  std::vector<std::vector<unsigned int>> dist_j(n_obs);
  double dist {};
  unsigned int vec_length_i {};
  arma::Col<unsigned int> vec_intervals(n_obs + 1);
  vec_intervals.at(0) = 0;
  std::size_t mat_size {0};
  if(haversine) {
    #pragma omp parallel for private(dist, vec_length_i) schedule(dynamic) reduction(+:mat_size) num_threads(n_cores)
    for(unsigned long int i = 0; i < n_obs; i++) {
      for(unsigned long int j = i + 1; j < n_obs; j++) {
        dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          dist = 1.0 - dist / dist_cutoff;
          dist_v[i].push_back(dist);
          dist_j[i].push_back(j);
        }
      }
      vec_length_i = dist_j[i].size();
      vec_intervals.at(i + 1) = vec_length_i;
      mat_size += vec_length_i;
    }
  } else {
    #pragma omp parallel for private(dist, vec_length_i) schedule(dynamic) reduction(+:mat_size) num_threads(n_cores)
    for(unsigned long int i = 0; i < n_obs; i++) {
      for(unsigned long int j = i + 1; j < n_obs; j++) {
        dist = euclidean_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          dist = 1.0 - dist / dist_cutoff;
          dist_v[i].push_back(dist);
          dist_j[i].push_back(j);
        }
      }
      vec_length_i = dist_j[i].size();
      vec_intervals.at(i + 1) = vec_length_i;
      mat_size += vec_length_i;
    }
  }
  if(mat_size < 4294967296) {
    vec_intervals = arma::cumsum(vec_intervals);
    arma::umat index_mat(2, mat_size);
    arma::vec values_vec(mat_size);
    unsigned int vec_begins_i {};
    unsigned int k {};
    if(batch_ram_opt > 2) {
      #pragma omp parallel for private(k, vec_begins_i, vec_length_i) schedule(dynamic) num_threads(n_cores)
      for(unsigned int i = 0; i < n_obs; i++) {
        vec_begins_i = vec_intervals.at(i);
        vec_length_i = vec_intervals.at(i + 1) - vec_begins_i;
        for(unsigned int j = 0; j < vec_length_i; j++) {
          k = vec_begins_i + j;
          index_mat.at(0, k) = i;
          index_mat.at(1, k) = dist_j[i][j];
          values_vec.at(k) = dist_v[i][j];
        }
        std::vector<double>().swap(dist_v[i]);
        std::vector<unsigned int>().swap(dist_j[i]);
      }
      vec_intervals.reset();
    } else {
      #pragma omp parallel for private(k, vec_begins_i, vec_length_i) schedule(dynamic) num_threads(n_cores)
      for(unsigned int i = 0; i < n_obs; i++) {
        vec_begins_i = vec_intervals.at(i);
        vec_length_i = vec_intervals.at(i + 1) - vec_begins_i;
        for(unsigned int j = 0; j < vec_length_i; j++) {
          k = vec_begins_i + j;
          index_mat.at(0, k) = i;
          index_mat.at(1, k) = dist_j[i][j];
          values_vec.at(k) = dist_v[i][j];
        }
      }
      if(batch_ram_opt > 1) {
        vec_intervals.reset();
        std::vector<std::vector<double>>().swap(dist_v);
        std::vector<std::vector<unsigned int>>().swap(dist_j);
      }
    }
    arma::sp_mat distances(index_mat, values_vec, n_obs, n_obs);
    distances.diag().ones();
    distances = arma::symmatu(distances);
    arma::mat f = lp_filling_s_d(distances, X, e, n_obs, n_vars, n_cores);
    return f;
  } else {
    arma::Col<unsigned long long int> vec_intervals_l = arma::cumsum(arma::conv_to<arma::Col<unsigned long long int>>::from(vec_intervals));
    if(batch_ram_opt > 1) {
      vec_intervals.reset();
    }
    arma::umat index_mat(2, mat_size);
    arma::vec values_vec(mat_size);
    unsigned long long int vec_begins_i {};
    unsigned long long int k {};
    if(batch_ram_opt > 2) {
      #pragma omp parallel for private(k, vec_begins_i, vec_length_i) schedule(dynamic) num_threads(n_cores)
      for(unsigned int i = 0; i < n_obs; i++) {
        vec_begins_i = vec_intervals_l.at(i);
        vec_length_i = vec_intervals_l.at(i + 1) - vec_begins_i;
        for(unsigned int j = 0; j < vec_length_i; j++) {
          k = vec_begins_i + j;
          index_mat.at(0, k) = i;
          index_mat.at(1, k) = dist_j[i][j];
          values_vec.at(k) = dist_v[i][j];
        }
        std::vector<double>().swap(dist_v[i]);
        std::vector<unsigned int>().swap(dist_j[i]);
      }
      vec_intervals_l.reset();
    } else {
      #pragma omp parallel for private(k, vec_begins_i, vec_length_i) schedule(dynamic) num_threads(n_cores)
      for(unsigned int i = 0; i < n_obs; i++) {
        vec_begins_i = vec_intervals_l.at(i);
        vec_length_i = vec_intervals_l.at(i + 1) - vec_begins_i;
        for(unsigned int j = 0; j < vec_length_i; j++) {
          k = vec_begins_i + j;
          index_mat.at(0, k) = i;
          index_mat.at(1, k) = dist_j[i][j];
          values_vec.at(k) = dist_v[i][j];
        }
      }
      if(batch_ram_opt > 1) {
        vec_intervals_l.reset();
        std::vector<std::vector<double>>().swap(dist_v);
        std::vector<std::vector<unsigned int>>().swap(dist_j);
      }
    }
    arma::sp_mat distances(index_mat, values_vec, n_obs, n_obs);
    distances.diag().ones();
    distances = arma::symmatu(distances);
    arma::mat f = lp_filling_s_d(distances, X, e, n_obs, n_vars, n_cores);
    return f;
  }
}

// 5 Function parallely calculating sparse distance matrix with a bartlett kernel and batch insert using floats and filling of spatial sandwich in logit and probit case
// [[Rcpp::export]]
arma::mat lp_f_b_p(arma::mat &coords, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_vars, double dist_cutoff, bool haversine,
  unsigned short int batch_ram_opt, unsigned int n_cores) {
  std::vector<std::vector<float>> dist_v(n_obs);
  std::vector<std::vector<unsigned int>> dist_j(n_obs);
  double dist {};
  float dist_f {};
  unsigned int vec_length_i {};
  arma::Col<unsigned int> vec_intervals(n_obs + 1);
  vec_intervals.at(0) = 0;
  std::size_t mat_size {0};
  if(haversine) {
    #pragma omp parallel for private(dist, dist_f, vec_length_i) schedule(dynamic) reduction(+:mat_size) num_threads(n_cores)
    for(unsigned long int i = 0; i < n_obs; i++) {
      for(unsigned long int j = i + 1; j < n_obs; j++) {
        dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          dist_f = (float) (1.0 - dist / dist_cutoff);
          dist_v[i].push_back(dist_f);
          dist_j[i].push_back(j);
        }
      }
      vec_length_i = dist_j[i].size();
      vec_intervals.at(i + 1) = vec_length_i;
      mat_size += vec_length_i;
    }
  } else {
    #pragma omp parallel for private(dist, dist_f, vec_length_i) schedule(dynamic) reduction(+:mat_size) num_threads(n_cores)
    for(unsigned long int i = 0; i < n_obs; i++) {
      for(unsigned long int j = i + 1; j < n_obs; j++) {
        dist = euclidean_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          dist_f = (float) (1.0 - dist / dist_cutoff);
          dist_v[i].push_back(dist_f);
          dist_j[i].push_back(j);
        }
      }
      vec_length_i = dist_j[i].size();
      vec_intervals.at(i + 1) = vec_length_i;
      mat_size += vec_length_i;
    }
  }
  if(mat_size < 4294967296) {
    vec_intervals = arma::cumsum(vec_intervals);
    arma::umat index_mat(2, mat_size);
    arma::fvec values_vec(mat_size);
    unsigned int vec_begins_i {};
    unsigned int k {};
    if(batch_ram_opt > 2) {
      #pragma omp parallel for private(k, vec_begins_i, vec_length_i) schedule(dynamic) num_threads(n_cores)
      for(unsigned int i = 0; i < n_obs; i++) {
        vec_begins_i = vec_intervals.at(i);
        vec_length_i = vec_intervals.at(i + 1) - vec_begins_i;
        for(unsigned int j = 0; j < vec_length_i; j++) {
          k = vec_begins_i + j;
          index_mat.at(0, k) = i;
          index_mat.at(1, k) = dist_j[i][j];
          values_vec.at(k) = dist_v[i][j];
        }
        std::vector<float>().swap(dist_v[i]);
        std::vector<unsigned int>().swap(dist_j[i]);
      }
      vec_intervals.reset();
    } else {
      #pragma omp parallel for private(k, vec_begins_i, vec_length_i) schedule(dynamic) num_threads(n_cores)
      for(unsigned int i = 0; i < n_obs; i++) {
        vec_begins_i = vec_intervals.at(i);
        vec_length_i = vec_intervals.at(i + 1) - vec_begins_i;
        for(unsigned int j = 0; j < vec_length_i; j++) {
          k = vec_begins_i + j;
          index_mat.at(0, k) = i;
          index_mat.at(1, k) = dist_j[i][j];
          values_vec.at(k) = dist_v[i][j];
        }
      }
      if(batch_ram_opt > 1) {
        vec_intervals.reset();
        std::vector<std::vector<float>>().swap(dist_v);
        std::vector<std::vector<unsigned int>>().swap(dist_j);
      }
    }
    arma::sp_fmat distances(index_mat, values_vec, n_obs, n_obs);
    distances.diag().ones();
    distances = arma::symmatu(distances);
    arma::mat f = lp_filling_s_f(distances, X, e, n_obs, n_vars, n_cores);
    return f;
  } else {
    arma::Col<unsigned long long int> vec_intervals_l = arma::cumsum(arma::conv_to<arma::Col<unsigned long long int>>::from(vec_intervals));
    if(batch_ram_opt > 1) {
      vec_intervals.reset();
    }
    arma::umat index_mat(2, mat_size);
    arma::fvec values_vec(mat_size);
    unsigned long long int vec_begins_i {};
    unsigned long long int k {};
    if(batch_ram_opt > 2) {
      #pragma omp parallel for private(k, vec_begins_i, vec_length_i) schedule(dynamic) num_threads(n_cores)
      for(unsigned int i = 0; i < n_obs; i++) {
        vec_begins_i = vec_intervals_l.at(i);
        vec_length_i = vec_intervals_l.at(i + 1) - vec_begins_i;
        for(unsigned int j = 0; j < vec_length_i; j++) {
          k = vec_begins_i + j;
          index_mat.at(0, k) = i;
          index_mat.at(1, k) = dist_j[i][j];
          values_vec.at(k) = dist_v[i][j];
        }
        std::vector<float>().swap(dist_v[i]);
        std::vector<unsigned int>().swap(dist_j[i]);
      }
      vec_intervals_l.reset();
    } else {
      #pragma omp parallel for private(k, vec_begins_i, vec_length_i) schedule(dynamic) num_threads(n_cores)
      for(unsigned int i = 0; i < n_obs; i++) {
        vec_begins_i = vec_intervals_l.at(i);
        vec_length_i = vec_intervals_l.at(i + 1) - vec_begins_i;
        for(unsigned int j = 0; j < vec_length_i; j++) {
          k = vec_begins_i + j;
          index_mat.at(0, k) = i;
          index_mat.at(1, k) = dist_j[i][j];
          values_vec.at(k) = dist_v[i][j];
        }
      }
      if(batch_ram_opt > 1) {
        vec_intervals_l.reset();
        std::vector<std::vector<float>>().swap(dist_v);
        std::vector<std::vector<unsigned int>>().swap(dist_j);
      }
    }
    arma::sp_fmat distances(index_mat, values_vec, n_obs, n_obs);
    distances.diag().ones();
    distances = arma::symmatu(distances);
    arma::mat f = lp_filling_s_f(distances, X, e, n_obs, n_vars, n_cores);
    return f;
  }
}

// 6 Function calculating sparse distance matrix with a uniform kernel and batch insert and filling of spatial sandwich in logit and probit case
// [[Rcpp::export]]
arma::mat lp_s_b(arma::mat &coords, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_vars, double dist_cutoff, bool haversine,
  unsigned short int batch_ram_opt) {
  std::vector<short int> dist_v;
  std::vector<std::vector<unsigned int>> dist_j(n_obs);
  double dist {};
  if(haversine) {
    for(unsigned long int i {0}; i < n_obs; i++) {
      for(unsigned long int j = i + 1; j < n_obs; j++) {
        dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          dist_v.push_back(1);
          dist_j[i].push_back(j);
        }
      }
    }
  } else {
    for(unsigned long int i {0}; i < n_obs; i++) {
      for(unsigned long int j = i + 1; j < n_obs; j++) {
        dist = euclidean_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          dist_v.push_back(1);
          dist_j[i].push_back(j);
        }
      }
    }
  }
  arma::umat index_mat(2, dist_v.size());
  arma::Col<short int> values_vec(dist_v);
  if(batch_ram_opt > 1) {
    std::vector<short int>().swap(dist_v);
  }
  std::size_t k {0};
  unsigned int vec_length_i {};
  for(unsigned int i {0}; i < n_obs; i++) {
    vec_length_i = dist_j[i].size();
    for(std::size_t j {0}; j < vec_length_i; j++) {
      index_mat.at(0, k) = i;
      index_mat.at(1, k) = dist_j[i][j];
      k++;
    }
  }
  if(batch_ram_opt > 1) {
    std::vector<std::vector<unsigned int>>().swap(dist_j);
  }
  arma::SpMat<short int> distances(index_mat, values_vec, n_obs, n_obs);
  distances.diag().ones();
  distances = arma::symmatu(distances);
  arma::mat f = lp_filling_s_s(distances, X, e, n_obs, n_vars, 1);
  return f;
}

// 7 Function calculating sparse distance matrix with a uniform kernel and batch insert and filling of spatial sandwich in logit and probit case
// [[Rcpp::export]]
arma::mat lp_s_b_p(arma::mat &coords, arma::mat &X, arma::vec &e, unsigned int n_obs, unsigned int n_vars, double dist_cutoff, bool haversine,
  unsigned short int batch_ram_opt, unsigned int n_cores) {
  std::vector<std::vector<short int>> dist_v(n_obs);
  std::vector<std::vector<unsigned int>> dist_j(n_obs);
  double dist {};
  unsigned int vec_length_i {};
  arma::Col<unsigned int> vec_intervals(n_obs + 1);
  vec_intervals.at(0) = 0;
  std::size_t mat_size {0};
  if(haversine) {
    #pragma omp parallel for private(dist, vec_length_i) schedule(dynamic) reduction(+:mat_size) num_threads(n_cores)
    for(unsigned long int i = 0; i < n_obs; i++) {
      for(unsigned long int j = i + 1; j < n_obs; j++) {
        dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          dist_v[i].push_back(1);
          dist_j[i].push_back(j);
        }
      }
      vec_length_i = dist_j[i].size();
      vec_intervals.at(i + 1) = vec_length_i;
      mat_size += vec_length_i;
    }
  } else {
    #pragma omp parallel for private(dist, vec_length_i) schedule(dynamic) reduction(+:mat_size) num_threads(n_cores)
    for(unsigned long int i = 0; i < n_obs; i++) {
      for(unsigned long int j = i + 1; j < n_obs; j++) {
        dist = euclidean_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          dist_v[i].push_back(1);
          dist_j[i].push_back(j);
        }
      }
      vec_length_i = dist_j[i].size();
      vec_intervals.at(i + 1) = vec_length_i;
      mat_size += vec_length_i;
    }
  }
  if(mat_size < 4294967296) {
    vec_intervals = arma::cumsum(vec_intervals);
    arma::umat index_mat(2, mat_size);
    arma::Col<short int> values_vec(mat_size);
    unsigned int vec_begins_i {};
    unsigned int k {};
    if(batch_ram_opt > 2) {
      #pragma omp parallel for private(k, vec_begins_i, vec_length_i) schedule(dynamic) num_threads(n_cores)
      for(unsigned int i = 0; i < n_obs; i++) {
        vec_begins_i = vec_intervals.at(i);
        vec_length_i = vec_intervals.at(i + 1) - vec_begins_i;
        for(unsigned int j = 0; j < vec_length_i; j++) {
          k = vec_begins_i + j;
          index_mat.at(0, k) = i;
          index_mat.at(1, k) = dist_j[i][j];
          values_vec.at(k) = dist_v[i][j];
        }
        std::vector<short int>().swap(dist_v[i]);
        std::vector<unsigned int>().swap(dist_j[i]);
      }
      vec_intervals.reset();
    } else {
      #pragma omp parallel for private(k, vec_begins_i, vec_length_i) schedule(dynamic) num_threads(n_cores)
      for(unsigned int i = 0; i < n_obs; i++) {
        vec_begins_i = vec_intervals.at(i);
        vec_length_i = vec_intervals.at(i + 1) - vec_begins_i;
        for(unsigned int j = 0; j < vec_length_i; j++) {
          k = vec_begins_i + j;
          index_mat.at(0, k) = i;
          index_mat.at(1, k) = dist_j[i][j];
          values_vec.at(k) = dist_v[i][j];
        }
      }
      if(batch_ram_opt > 1) {
        vec_intervals.reset();
        std::vector<std::vector<short int>>().swap(dist_v);
        std::vector<std::vector<unsigned int>>().swap(dist_j);
      }
    }
    arma::SpMat<short int> distances(index_mat, values_vec, n_obs, n_obs);
    distances.diag().ones();
    distances = arma::symmatu(distances);
    arma::mat f = lp_filling_s_s(distances, X, e, n_obs, n_vars, n_cores);
    return f;
  } else {
    arma::Col<unsigned long long int> vec_intervals_l = arma::cumsum(arma::conv_to<arma::Col<unsigned long long int>>::from(vec_intervals));
    if(batch_ram_opt > 1) {
      vec_intervals.reset();
    }
    arma::umat index_mat(2, mat_size);
    arma::Col<short int> values_vec(mat_size);
    unsigned long long int vec_begins_i {};
    unsigned long long int k {};
    if(batch_ram_opt > 2) {
      #pragma omp parallel for private(k, vec_begins_i, vec_length_i) schedule(dynamic) num_threads(n_cores)
      for(unsigned int i = 0; i < n_obs; i++) {
        vec_begins_i = vec_intervals_l.at(i);
        vec_length_i = vec_intervals_l.at(i + 1) - vec_begins_i;
        for(unsigned int j = 0; j < vec_length_i; j++) {
          k = vec_begins_i + j;
          index_mat.at(0, k) = i;
          index_mat.at(1, k) = dist_j[i][j];
          values_vec.at(k) = dist_v[i][j];
        }
        std::vector<short int>().swap(dist_v[i]);
        std::vector<unsigned int>().swap(dist_j[i]);
      }
      vec_intervals_l.reset();
    } else {
      #pragma omp parallel for private(k, vec_begins_i, vec_length_i) schedule(dynamic) num_threads(n_cores)
      for(unsigned int i = 0; i < n_obs; i++) {
        vec_begins_i = vec_intervals_l.at(i);
        vec_length_i = vec_intervals_l.at(i + 1) - vec_begins_i;
        for(unsigned int j = 0; j < vec_length_i; j++) {
          k = vec_begins_i + j;
          index_mat.at(0, k) = i;
          index_mat.at(1, k) = dist_j[i][j];
          values_vec.at(k) = dist_v[i][j];
        }
      }
      if(batch_ram_opt > 1) {
        vec_intervals_l.reset();
        std::vector<std::vector<short int>>().swap(dist_v);
        std::vector<std::vector<unsigned int>>().swap(dist_j);
      }
    }
    arma::SpMat<short int> distances(index_mat, values_vec, n_obs, n_obs);
    distances.diag().ones();
    distances = arma::symmatu(distances);
    arma::mat f = lp_filling_s_s(distances, X, e, n_obs, n_vars, n_cores);
    return f;
  }
}


