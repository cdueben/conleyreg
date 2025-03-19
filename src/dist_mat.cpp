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
#include <vector>
#include <cstddef>
#include "distance_functions.h"

// 1 dist_mat_d_d Function computing dense distance matrix with distance cutoff
// 2 dist_mat_d Function computing dense distance matrix without distance cutoff
// 3 dist_spmat_d_d Function computing sparse distance matrix with distance cutoff
// 4 dist_spmat_d_d_b Function computing sparse distance matrix with batch insert and distance cutoff
// 5 dist_spmat_d_d_b_p Function parallely computing sparse distance matrix with batch insert and distance cutoff
// 6 dist_spmat_d_d_r Function computing sparse matrix of rounded distances with distance cutoff
// 7 dist_spmat_d_d_b_r Function computing sparse matrix of rounded distances with batch insert and distance cutoff
// 8 dist_spmat_d_d_b_r_p Function parallely computing sparse matrix of rounded distances with batch insert and distance cutoff

// 1 Function computing dense distance matrix with distance cutoff
// [[Rcpp::export]]
arma::mat dist_mat_d_d(arma::mat &coords, unsigned int n_obs, double dist_cutoff, bool haversine, unsigned short int n_cores) {
  arma::vec v_nan(n_obs, arma::fill::value(arma::datum::nan));
  arma::mat distances = arma::diagmat(v_nan);
  v_nan.reset();
  double dist {};
  double z {0.0};
  if(haversine) {
    #pragma omp parallel for private(dist) schedule(dynamic) num_threads(n_cores) if(n_cores > 1)
    for(unsigned int i = 0; i < n_obs; i++) {
      for(unsigned int j = i + 1; j < n_obs; j++) {
        dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          if(dist == z) {
            dist = arma::datum::nan;
          }
          distances.at(i, j) = dist;
          distances.at(j, i) = dist;
        }
      }
    }
  } else {
    #pragma omp parallel for private(dist) schedule(dynamic) num_threads(n_cores) if(n_cores > 1)
    for(unsigned int i = 0; i < n_obs; i++) {
      for(unsigned int j = i + 1; j < n_obs; j++) {
        dist = euclidean_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          if(dist == z) {
            dist = arma::datum::nan;
          }
          distances.at(i, j) = dist;
          distances.at(j, i) = dist;
        }
      }
    }
  }
  return distances;
}

// 2 Function computing dense distance matrix without distance cutoff
// [[Rcpp::export]]
arma::mat dist_mat_d(arma::mat &coords, unsigned int n_obs, bool haversine, unsigned short int n_cores) {
  arma::mat distances(n_obs, n_obs, arma::fill::zeros);
  double dist {};
  if(haversine) {
    #pragma omp parallel for private(dist) schedule(dynamic) num_threads(n_cores) if(n_cores > 1)
    for(unsigned int i = 0; i < n_obs; i++) {
      for(unsigned int j = i + 1; j < n_obs; j++) {
        dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        distances.at(i, j) = dist;
        distances.at(j, i) = dist;
      }
    }
  } else {
    #pragma omp parallel for private(dist) schedule(dynamic) num_threads(n_cores) if(n_cores > 1)
    for(unsigned int i = 0; i < n_obs; i++) {
      for(unsigned int j = i + 1; j < n_obs; j++) {
        dist = euclidean_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        distances.at(i, j) = dist;
        distances.at(j, i) = dist;
      }
    }
  }
  return distances;
}

// 3 Function computing sparse distance matrix with distance cutoff
// [[Rcpp::export]]
arma::sp_mat dist_spmat_d_d(arma::mat &coords, unsigned int n_obs, double dist_cutoff, bool haversine, unsigned short int n_cores) {
  arma::sp_mat distances(n_obs, n_obs);
  double dist {};
  double z {0.0};
  if(haversine) {
    #pragma omp parallel for private(dist) schedule(dynamic) num_threads(n_cores) if(n_cores > 1)
    for(unsigned int i = 0; i < n_obs; i++) {
      distances.at(i, i) = arma::datum::nan;
      for(unsigned int j = i + 1; j < n_obs; j++) {
        dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          if(dist == z) {
            dist = arma::datum::nan;
          }
          distances.at(i, j) = dist;
        }
      }
    }
  } else {
    #pragma omp parallel for private(dist) schedule(dynamic) num_threads(n_cores) if(n_cores > 1)
    for(unsigned int i = 0; i < n_obs; i++) {
      distances.at(i, i) = arma::datum::nan;
      for(unsigned int j = i + 1; j < n_obs; j++) {
        dist = euclidean_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          if(dist == z) {
            dist = arma::datum::nan;
          }
          distances.at(i, j) = dist;
        }
      }
    }
  }
  distances = arma::symmatu(distances);
  return distances;
}

// 4 Function computing sparse distance matrix with batch insert and distance cutoff
// [[Rcpp::export]]
arma::sp_mat dist_spmat_d_d_b(arma::mat &coords, unsigned int n_obs, double dist_cutoff, bool haversine, unsigned short int batch_ram_opt) {
  std::vector<double> dist_v;
  std::vector<std::vector<unsigned int>> dist_j(n_obs);
  double dist {};
  double z {0.0};
  if(haversine) {
    for(unsigned int i {0}; i < n_obs; i++) {
      for(unsigned int j = i + 1; j < n_obs; j++) {
        dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          if(dist == z) {
            dist = arma::datum::nan;
          }
          dist_v.push_back(dist);
          dist_j[i].push_back(j);
        }
      }
    }
  } else {
    for(unsigned int i {0}; i < n_obs; i++) {
      for(unsigned int j = i + 1; j < n_obs; j++) {
        dist = euclidean_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          if(dist == z) {
            dist = arma::datum::nan;
          }
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
  distances.diag() = arma::vec(n_obs, arma::fill::value(arma::datum::nan));
  distances = arma::symmatu(distances);
  return distances;
}

// 5 Function parallely computing distance sparse matrix with batch insert and distance cutoff
// [[Rcpp::export]]
arma::sp_mat dist_spmat_d_d_b_p(arma::mat &coords, unsigned int n_obs, double dist_cutoff, bool haversine, unsigned short int batch_ram_opt,
  unsigned short int n_cores) {
  std::vector<std::vector<double>> dist_v(n_obs);
  std::vector<std::vector<unsigned int>> dist_j(n_obs);
  double dist {};
  double z {0.0};
  unsigned int vec_length_i {};
  arma::Col<unsigned int> vec_intervals(n_obs + 1);
  vec_intervals.at(0) = 0;
  std::size_t mat_size {0};
  if(haversine) {
    #pragma omp parallel for private(dist, vec_length_i) schedule(dynamic) reduction(+:mat_size) num_threads(n_cores)
    for(unsigned int i = 0; i < n_obs; i++) {
      for(unsigned int j = i + 1; j < n_obs; j++) {
        dist = haversine_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          if(dist == z) {
            dist = arma::datum::nan;
          }
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
    for(unsigned int i = 0; i < n_obs; i++) {
      for(unsigned int j = i + 1; j < n_obs; j++) {
        dist = euclidean_dist(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          if(dist == z) {
            dist = arma::datum::nan;
          }
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
    distances.diag() = arma::vec(n_obs, arma::fill::value(arma::datum::nan));
    distances = arma::symmatu(distances);
    return distances;
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
    distances.diag() = arma::vec(n_obs, arma::fill::value(arma::datum::nan));
    distances = arma::symmatu(distances);
    return distances;
  }
}

// 6 Function computing sparse matrix of rounded distances with distance cutoff
// [[Rcpp::export]]
arma::SpMat<unsigned int> dist_spmat_d_d_r(arma::mat &coords, unsigned int n_obs, unsigned int dist_cutoff, bool haversine, unsigned short int n_cores) {
  arma::SpMat<unsigned int> distances(n_obs, n_obs);
  unsigned int dist {};
  unsigned int z {0};
  if(haversine) {
    #pragma omp parallel for private(dist) schedule(dynamic) num_threads(n_cores) if(n_cores > 1)
    for(unsigned int i = 0; i < n_obs; i++) {
      distances.at(i, i) = arma::datum::nan;
      for(unsigned int j = i + 1; j < n_obs; j++) {
        dist = haversine_dist_r(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          if(dist == z) {
            dist = arma::datum::nan;
          }
          distances.at(i, j) = dist;
        }
      }
    }
  } else {
    #pragma omp parallel for private(dist) schedule(dynamic) num_threads(n_cores) if(n_cores > 1)
    for(unsigned int i = 0; i < n_obs; i++) {
      distances.at(i, i) = arma::datum::nan;
      for(unsigned int j = i + 1; j < n_obs; j++) {
        dist = euclidean_dist_r(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          if(dist == z) {
            dist = arma::datum::nan;
          }
          distances.at(i, j) = dist;
        }
      }
    }
  }
  distances = arma::symmatu(distances);
  return distances;
}

// 7 Function computing sparse matrix of rounded distances with batch insert and distance cutoff
// [[Rcpp::export]]
arma::SpMat<unsigned int> dist_spmat_d_d_b_r(arma::mat &coords, unsigned int n_obs, unsigned int dist_cutoff, bool haversine, unsigned short int batch_ram_opt) {
  std::vector<unsigned int> dist_v;
  std::vector<std::vector<unsigned int>> dist_j(n_obs);
  unsigned int dist {};
  unsigned int z {0};
  if(haversine) {
    for(unsigned int i {0}; i < n_obs; i++) {
      for(unsigned int j = i + 1; j < n_obs; j++) {
        dist = haversine_dist_r(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          if(dist == z) {
            dist = arma::datum::nan;
          }
          dist_v.push_back(dist);
          dist_j[i].push_back(j);
        }
      }
    }
  } else {
    for(unsigned int i {0}; i < n_obs; i++) {
      for(unsigned int j = i + 1; j < n_obs; j++) {
        dist = euclidean_dist_r(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          if(dist == z) {
            dist = arma::datum::nan;
          }
          dist_v.push_back(dist);
          dist_j[i].push_back(j);
        }
      }
    }
  }
  arma::umat index_mat(2, dist_v.size());
  arma::Col<unsigned int> values_vec(dist_v);
  if(batch_ram_opt > 1) {
    std::vector<unsigned int>().swap(dist_v);
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
  arma::SpMat<unsigned int> distances(index_mat, values_vec, n_obs, n_obs);
  distances.diag() = arma::Col<unsigned int>(n_obs, arma::fill::value(arma::datum::nan));
  distances = arma::symmatu(distances);
  return distances;
}

// 8 Function parallely computing sparse matrix of rounded distances with batch insert and distance cutoff
// [[Rcpp::export]]
arma::SpMat<unsigned int> dist_spmat_d_d_b_r_p(arma::mat &coords, unsigned int n_obs, unsigned int dist_cutoff, bool haversine, unsigned short int batch_ram_opt,
  unsigned short int n_cores) {
  std::vector<std::vector<unsigned int>> dist_v(n_obs);
  std::vector<std::vector<unsigned int>> dist_j(n_obs);
  unsigned int dist {};
  unsigned int z {0};
  unsigned int vec_length_i {};
  arma::Col<unsigned int> vec_intervals(n_obs + 1);
  vec_intervals.at(0) = 0;
  std::size_t mat_size {0};
  if(haversine) {
    #pragma omp parallel for private(dist, vec_length_i) schedule(dynamic) reduction(+:mat_size) num_threads(n_cores)
    for(unsigned int i = 0; i < n_obs; i++) {
      for(unsigned int j = i + 1; j < n_obs; j++) {
        dist = haversine_dist_r(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          if(dist == z) {
            dist = arma::datum::nan;
          }
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
    for(unsigned int i = 0; i < n_obs; i++) {
      for(unsigned int j = i + 1; j < n_obs; j++) {
        dist = euclidean_dist_r(coords(i, 1), coords(j, 1), coords(i, 0), coords(j, 0));
        if(dist < dist_cutoff) {
          if(dist == z) {
            dist = arma::datum::nan;
          }
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
    arma::Col<unsigned int> values_vec(mat_size);
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
        std::vector<unsigned int>().swap(dist_v[i]);
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
        std::vector<std::vector<unsigned int>>().swap(dist_v);
        std::vector<std::vector<unsigned int>>().swap(dist_j);
      }
    }
    arma::SpMat<unsigned int> distances(index_mat, values_vec, n_obs, n_obs);
    distances.diag() = arma::Col<unsigned int>(n_obs, arma::fill::value(arma::datum::nan));
    distances = arma::symmatu(distances);
    return distances;
  } else {
    arma::Col<unsigned long long int> vec_intervals_l = arma::cumsum(arma::conv_to<arma::Col<unsigned long long int>>::from(vec_intervals));
    if(batch_ram_opt > 1) {
      vec_intervals.reset();
    }
    arma::umat index_mat(2, mat_size);
    arma::Col<unsigned int> values_vec(mat_size);
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
        std::vector<unsigned int>().swap(dist_v[i]);
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
        std::vector<std::vector<unsigned int>>().swap(dist_v);
        std::vector<std::vector<unsigned int>>().swap(dist_j);
      }
    }
    arma::SpMat<unsigned int> distances(index_mat, values_vec, n_obs, n_obs);
    distances.diag() = arma::Col<unsigned int>(n_obs, arma::fill::value(arma::datum::nan));
    distances = arma::symmatu(distances);
    return distances;
  }
}
