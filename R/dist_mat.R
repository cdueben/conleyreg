#' Distance matrix estimation
#'
#' This function estimates the distance matrix separately from Conley standard errors. Such step can be helpful when running multiple Conley standard error estimations
#' based on the same distance matrix. A pre-requisite of using this function is that the data must not be modified between applying this function and inserting the
#' results into \code{conleyreg}.
#'
#' @param data input data. Either (i) in non-spatial data frame format (includes tibbles and data tables) with columns denoting coordinates or (ii) in sf format. In
#' case of an sf object, all non-point geometry types are converted to spatial points, based on the feature's centroid. When using a non-spatial data frame format
#' the with projected, i.e. non-longlat, coordinates, \code{crs} must be specified. Note that the projection can influence the computed distances, which is a general
#' phenomenon in GIS software and not specific to conleyreg. The computationally fastest option is to use a data table with coordinates in the crs in which
#' the distances are to be derived (longlat for spherical and projected for planar), and with time and unit set as keys in the panel case. An sf object as input is
#' the slowest option.
#' @param unit the variable identifying the cross-sectional dimension. Only needs to be specified, if data is not cross-sectional. Assumes that units do not change
#' their location over time.
#' @param time the variable identifying the time dimension. Only needs to be specified, if data is not cross-sectional.
#' @param lat the variable specifying the latitude
#' @param lon the variable specifying the longitude
#' @param dist_comp choice between \code{spherical} and \code{planar} distance computations. When unspecified, the input data determines the method: longlat uses
#' spherical (Haversine) distances, alternatives (projected data) use planar (Euclidean) distances. When inserting projected data but specifying
#' \code{dist_comp = "spherical"}, the data is transformed to longlat. Combining unprojected data with \code{dist_comp = "planar"} transforms the data to an
#' azimuthal equidistant format centered at the data's centroid.
#' @param dist_cutoff the distance cutoff in km. If not specified, the distances matrices contain all bilateral distances. If specified, the cutoff most be as least
#' as large as the largest distance cutoff in the Conley standard error corrections in which you use the resulting matrix. If you e.g. specify distance cutoffs of
#' 100, 200, and 500 km in the subsequent \code{conleyreg} calls, \code{dist_cutoff} in this function must be set to at least 500. \code{dist_cutoff}
#' allows to pre-compute distance matrices also in applications where a full distance matrix would not fit into the computer's memory - conditional on that
#' \code{sparse = TRUE}.
#' @param crs the coordinate reference system, if the data is projected. Object of class crs or input string to \code{sf::st_crs}. This parameter can be omitted, if
#' the data is in longlat format (EPSG: 4326), i.e. not projected. If the projection does not use meters as units, this function converts to units to meters.
#' @param verbose logical specifying whether to print messages on intermediate estimation steps. Defaults to \code{TRUE}.
#' @param ncores the number of CPU cores to use in the estimations. Defaults to the machine's number of CPUs.
#' @param par_dim the dimension along which the function parallelizes in unbalanced panel applications. Can be set to \code{"cross-section"} (default) or
#' \code{"time"}. Use \code{"r"} and \code{"cpp"} to define parallelization based on the language rather than the dimension. In this function, \code{"r"} is
#' equivalent to \code{"time"} and parallelizes along the time dimension using the parallel package. \code{"cross-section"} is equivalent to \code{"cpp"} and
#' parallelizes along the cross-sectional dimension using OpenMP in C++. Some MAC users do not have access to OpenMP by default. \code{par_dim} is then always set to
#' \code{"r"}. Thus, depending on the application, the function can be notably faster on Windows and Linux than on MACs. When \code{st_distance = TRUE}, \code{par_dim}
#' defaults to \code{"time"}.
#' @param sparse logical specifying whether to use sparse rather than dense (regular) matrices in distance computations. Defaults to \code{FALSE}. Only has an effect
#' when \code{st_distance = FALSE}. Sparse matrices are more efficient than dense matrices, when the distance matrix has a lot of zeros arising from points located
#' outside the respective \code{dist_cutoff}. It is recommended to keep the default unless the machine is unable to allocate enough memory. The function always uses
#' dense matrices when \code{dist_cutoff} is not specified.
#' @param batch logical specifying whether distances are inserted into a sparse matrix element by element (\code{FALSE}) or all at once as a batch (\code{TRUE}).
#' Defaults to \code{FALSE}. This argument only has an effect when \code{st_distance = FALSE} and \code{sparse = TRUE}. Batch insertion is faster than element-wise
#' insertion, but requires more memory.
#' @param batch_ram_opt the degree to which batch insertion should be optimized for RAM usage. Can be set to one out of the three levels: \code{"none"},
#' \code{"moderate"} (default), and \code{"heavy"}. Higher levels imply lower RAM usage, but also lower speeds.
#' @param dist_round logical specifying whether to round distances to full kilometers. This further reduces memory consumption and can be a solution when even sparse
#' matrices cannot accomodate the data. Note, though, that this rounding introduces a bias.
#' @param st_distance logical specifying whether distances should be computed via \code{sf::st_distance} (\code{TRUE}) or via conleyreg's internal, computationally
#' optimized distance functions (\code{FALSE}). The default (\code{FALSE}) produces the same distances as \code{sf::st_distance} does with S2 enabled. I.e. it uses
#' Haversine (great circle) distances for longlat data and Euclidean distances otherwise. Cases in which you might want to set this argument to \code{TRUE} are e.g.
#' when you want enforce the GEOS approach to computing distances or when you are using a peculiar projection, for which the sf package might include further
#' procedures. Cross-sectional parallelization is not available when \code{st_distance = TRUE} and the function automatically switches to parallelization along the
#' time dimension, if the data is a panel and \code{ncores != 1}. Third and fourth dimensions, termed Z and M in sf, are not accounted for in any case. Note that
#' \code{sf::st_distance} is considerably slower than conleyreg's internal distance functions.
#' @param dist_which the type of distance to use when \code{st_distance = TRUE}. If unspecified, the function defaults to great circle distances for longlat data and
#' to Euclidean distances otherwise. See \code{sf::st_distance} for options.
#'
#' @details This function runs the distance matrix estimations separately from the Conley standard error correction. You can pass the resulting object to the
#' \code{dist_mat} argument in \code{conleyreg}, skipping the distance matrix computations and various checks in that function. Pre-computing the distance matrix
#' is only more efficient than deriving it via \code{conleyreg} when estimating various models that use the same distance matrices. The input data must not be
#' modified between calling this function and inserting the results into \code{conleyreg}. Do not reorder the observations, add or delete variables, or undertake
#' any other operation on the data.
#'
#' @return Returns an object of S3 class \code{conley_dist}. It contains modified distance matrices, the used \code{dist_cutoff}, a sparse matrix identifier, and
#' information on the potential panel structure. In the cross-sectional case and the balanced panel case, the distances are stored in one matrix, while in unbalanced
#' panel applications, distances come as a list of matrices. The function optimizes the distance matrices with respect to computational performance, setting
#' distances beyond \code{dist_cutoff} to zero and actual off-diagonal zeros to NaN. Hence, these objects are only to be used in \code{conleyreg}.
#'
#' @examples
#' \dontrun{
#' # Generate cross-sectional example data
#' data <- rnd_locations(100, output_type = "data.frame")
#' data$y <- sample(c(0, 1), 100, replace = TRUE)
#' data$x1 <- stats::runif(100, -50, 50)
#'
#' # Compute distance matrix in cross-sectional case
#' dm <- dist_mat(data, lat = "lat", lon = "lon")
#'
#' # Compute distance matrix in panel case
#' data$time <- rep(1:10, each = 10)
#' data$unit <- rep(1:10, times = 10)
#' dm <- dist_mat(data, unit = "unit", time = "time", lat = "lat", lon = "lon")
#'
#' # Use distance matrix in conleyreg function
#' conleyreg(y ~ x1, data, 1000, dist_mat = dm)
#' }
#'
#' @export
dist_mat <- function(data, unit = NULL, time = NULL, lat = NULL, lon = NULL, dist_comp = NULL, dist_cutoff = NULL, crs = NULL, verbose = TRUE, ncores = NULL,
  par_dim = c("cross-section", "time", "r", "cpp"), sparse = FALSE, batch = TRUE, batch_ram_opt = NULL, dist_round = FALSE, st_distance = FALSE, dist_which = NULL) {
  # Check whether data is panel
  panel <- !is.null(time)

  # Set parallel configuration
  if(is.null(ncores)) {
    ncores <- parallel::detectCores()
    if(openmp_installed()) {
      par_dim <- match.arg(par_dim)
    } else {
      par_dim <- "r"
    }
  } else if(is.numeric(ncores) && ncores > 0) {
    ncores <- as.integer(ncores)
    if(openmp_installed()) {
      par_dim <- match.arg(par_dim)
    } else {
      par_dim <- "r"
    }
  } else {
    stop("ncores must be either NULL or a positive integer")
  }

  # Check arguments
  if(panel) {
    if(length(unit) != 1 || !(unit %in% names(data))) stop("In panel applications, unit must be set to the name of variable in the data set")
    if(length(time) != 1 || !(time %in% names(data))) stop("In panel applications, time must be set to the name of variable in the data set")
  }
  if(!is.null(dist_comp) && !(dist_comp %in% c("spherical", "planar"))) {
    stop(paste0(dist_comp, ' is not a valid dist_comp value. Choose either NULL (unspecified), "spherical", or "planar".'))
  }
  if(!is.null(dist_cutoff) && (length(dist_cutoff) != 1 || !is.numeric(dist_cutoff) || dist_cutoff < 0)) {
    stop("dist_cutoff must be NULL or numeric, >= 0, and of length one")
  }
  if(length(verbose) != 1 || !is.logical(verbose) || is.na(verbose)) stop("verbose must be logical and of length one")
  if(length(sparse) != 1 || !is.logical(sparse) || is.na(sparse)) stop("sparse must be logical and of length one")
  if(length(batch) != 1 || !is.logical(batch) || is.na(batch)) stop("batch must be logical and of length one")
  if(length(dist_round) != 1 || !is.logical(dist_round) || is.na(dist_round)) stop("dist_round must be logical and of length one")
  if(length(st_distance) != 1 || !is.logical(st_distance) || is.na(st_distance)) stop("st_distance must be logical and of length one")
  batch_ram_opt <- which(match.arg(batch_ram_opt, c("moderate", "none", "heavy")) == c("none", "moderate", "heavy"))
  if(st_distance && par_dim %in% c("cross-section", "cpp") && ncores > 1L) {
    par_dim <- "time"
    if(verbose) {
      message('Parallelization along the cross-sectional dimension unavailable when st_distance = TRUE. Thus, par_dim changed to "time". Specify ncores = 1 to run ',
        'the code serially instead.')
    }
  }
  if(is.null(dist_cutoff) && sparse) stop("Use dense matrices when not specifying dist_cutoff")

  # Add indicator signaling copied data (if data was copied since function start, data table's set functions can be used without changing the input data)
  data_copied <- FALSE

  # Subset the data to variables that are used in the estimation
  if(panel) vars <- c(unit, time, lat, lon) else vars <- c(lat, lon)
  if(any(class(data) == "data.table")) {
    data <- data[, eval(vars), with = FALSE]
  } else {
    nrow_i <- NROW(data)
    ncol_i <- NCOL(data)
    data <- data[, vars]
    if(NCOL(data) < ncol_i) data_copied <- TRUE
  }

  # Check spatial attributes
  if(verbose) message("Checking spatial attributes")
  if(any(class(data) == "sf")) {
    # Check if the CRS is set
    if(is.na(sf::st_crs(data))) stop("CRS not set")
    # Convert to points
    if(!all(sf::st_geometry_type(data) == "POINT")) {
      data <- sf::st_centroid(data)
      data_copied <- TRUE
    }
    # Drop third and fourth dimensions in case of conleyreg distance computations
    data <- sf::st_zm(data)
    # Check if the CRS is either longlat or uses meters as units (otherwise convert units to meters)
    if(!sf::st_is_longlat(data)) {
      if(gsub("[+]units=", "", regmatches(sf::st_crs(data)$proj4string, regexpr("[+]units=+\\S", sf::st_crs(data)$proj4string))) != "m") {
        data <- sf::st_transform(data, crs = gsub("[+]units=+\\S", "+units=m", sf::st_crs(data)$proj4string))
        data_copied <- TRUE
      }
      longlat <- FALSE
    } else if(sf::st_is_longlat(data)) {
      longlat <- TRUE
    }
  } else if(any(class(data) == "data.frame")) {
    # Check if lat and lon are set
    if(is.null(lat) || is.null(lon)) stop("When data providing data in non-spatial format, you need to specify lat and lon")
    longlat <- (is.null(crs) || sf::st_is_longlat(sf::st_crs(crs)))
    if(longlat) {
      # Check if coordinates are non-missing and longitudes are between -180 and 180 and latitudes are between -90 and 90
      if(any(class(data) == "data.table")) {
        ymin <- min(data[[lat]])
        ymax <- max(data[[lat]])
        xmin <- min(data[[lon]])
        xmax <- max(data[[lon]])
      } else {
        ymin <- min(data[, lat, drop = TRUE])
        ymax <- max(data[, lat, drop = TRUE])
        xmin <- min(data[, lon, drop = TRUE])
        xmax <- max(data[, lon, drop = TRUE])
      }
      if(any(is.na(c(ymin, ymax, xmin, xmax)))) stop("Coordinates contain missing values")
      if(any(c(ymin, ymax) < -90) || any(c(ymin, ymax) > 90) || any(c(xmin, xmax) < -180) || any(c(xmin, xmax) > 180)) {
        longlat <- FALSE
      } else {
        longlat <- TRUE
      }
      rm(ymin, ymax, xmin, xmax)
    }
    if(!longlat) {
      if(is.null(crs)) {
        stop("crs must be specified when entering projected data")
      } else {
        crs <- sf::st_crs(crs)$proj4string
        if(gsub("[+]units=", "", regmatches(crs, regexpr("[+]units=+\\S", crs))) != "m") {
          data <- sf::st_transform(sf::st_as_sf(data, coords = c(lon, lat), crs = crs), crs = gsub("[+]units=+\\S", "+units=m", crs))
        }
      }
    }
  } else {
    stop(paste0("Data of class ", class(data), " is not a valid input"))
  }

  # Set dist_comp if unspecified
  if(is.null(dist_comp)) {
    if(longlat) dist_comp <- "spherical" else dist_comp <- "planar"
  }
  if(st_distance && !is.null(dist_which) && dist_comp == "spherical") warning('dist_which ignored as dist_comp = "spherical"')
  # Adjust projection
  if(!longlat && dist_comp == "spherical") {
    if(verbose) message("Converting projected data to longlat")
    if(!any(class(data) == "sf")) {
      data <- sf::st_as_sf(data, coords = c(lon, lat), crs = crs)
    }
    crs <- 4326
    data <- sf::st_transform(data, crs = crs)
    data_copied <- TRUE
  } else if(longlat && dist_comp == "planar") {
    if(!any(class(data) == "sf")) data <- sf::st_as_sf(data, coords = c(lon, lat), crs = 4326)
    warning("Projecting longlat data to an azimuthal equidistant format centered at the data's centroid. Note that with this projection, only connections ",
      "passing through the centroid result in correct distance estimates. Other distance estimates can be heavily biased. It is recommended to convert the data to ",
      "a projection that is appropriate for the specific application before inserting it into conleyreg, or to use spherical distance computations.")
    crs <- sf::st_coordinates(sf::st_centroid(sf::st_as_sfc(sf::st_bbox(data))))
    crs <- paste0("+proj=aeqd +lat_0=", crs[1, 2], " +lon_0=", crs[1, 1])
    data <- sf::st_transform(data, crs = crs)
    data_copied <- TRUE
  } else if(st_distance && is.null(crs)) {
    if(any(class(data) == "sf")) {
      crs <- sf::st_crs(data)
    } else {
      if(longlat) {
        crs <- 4326
      } else {
        stop("crs must by specified when entering projected data and using st_distance = TRUE")
      }
    }
  }

  # Convert data to data table, if distance computations are not done via st_distance or if data is a panel
  if(panel || !st_distance) {
    if(any(class(data) == "sf")) {
      lon <- "X"
      lat <- "Y"
      if(lon %in% names(data) || lat %in% names(data)) {
        i <- 1
        while(lon %in% names(data)) {
          lon <- paste0("X", i)
          i <- 1 + 1
        }
        i <- 1
        while(lat %in% names(data)) {
          lat <- paste0("Y", i)
          i <- 1 + 1
        }
        rm(i)
        data <- data.table::data.table(sf::st_drop_geometry(data), data.table::setnames(data.table::setDT(sf::st_coordinates(data)), c("X", "Y"), c(lon, lat)))
      } else {
        data <- data.table::data.table(sf::st_drop_geometry(data), sf::st_coordinates(data))
      }
      if(panel) data.table::setkeyv(data, time)
    } else if(panel) {
      if(any(class(data) == "data.table")) {
        data.table::setkeyv(data, time)
      } else {
        if(data_copied) {
          data.table::setDT(data, key = time)
        } else {
          data <- data.table::as.data.table(data, key = time)
        }
      }
    }
  }

  # Check if panel is balanced
  if(panel) {
    if(is.null(unit)) stop("Cross-sectional identifier, unit, not set")
    if(data.table::uniqueN(data, by = time) > 1) {
      balanced <- isbalancedcpp(as.matrix(data.table::setorderv(data[, eval(c(time, unit)), with = FALSE], c(time, unit))))
      if(balanced == 1) {
        balanced <- TRUE
      } else if(balanced == 0) {
        balanced <- FALSE
      } else { # balanced == 2
        stop(paste0(unit), " does not uniquely identify cross-sectional units")
      }
      if(!balanced && verbose) message("Unbalanced panel identified")
      # Obtain number of observations
      n_obs <- NROW(data)
      # Drop missing values
      data <- stats::na.omit(data)
      if(NROW(data) < n_obs && balanced) {
        balanced <- FALSE
        warning("Panel treated as unbalanced because of missing values")
      }
      if(balanced && verbose) message("Balanced panel identified")
    } else {
      panel <- FALSE
      warning("Number of time periods: 1. Treating data as cross-sectional.")
      # Drop missing values in cross-sectional case
      data <- stats::na.omit(data)
    }
    if(verbose) message("Computing distances")
    # Panel application
    if(balanced) {
      # Subset data to one period
      data <- data[.(data[1, time]), on = eval(time)]
      data.table::setorderv(data, unit)
      # Distance matrices are identical across time periods in a balanced panel
      distances <- dist_fun_dm(data[, eval(c(lon, lat)), with = FALSE], dist_comp, sparse, batch, batch_ram_opt, dist_cutoff, dist_round, crs, st_distance, dist_which,
        ncores)
    } else {
      # Unbalanced panel
      if(ncores > 1 && par_dim %in% c("time", "r")) {
        # Parallel computation along time dimension
        cl <- parallel::makePSOCKcluster(ncores)
        doParallel::registerDoParallel(cl)
        distances <- foreach::foreach(tp = sort(unique(data[[time]]))) %dopar% {
          distances_tp <- data[.(tp), on = eval(time)]
          data.table::setorderv(distances_tp, unit)
          distances_tp <- dist_fun_dm(distances_tp[, eval(c(lon, lat)), with = FALSE], dist_comp, sparse, batch, batch_ram_opt, dist_cutoff, dist_round, crs,
            st_distance, dist_which, 1L)
          return(distances_tp)
        }
        parallel::stopCluster(cl)
      } else {
        # Non-parallel computation or parallel compuation along cross-sectional dimension
        distances <- foreach::foreach(tp = sort(unique(data[[time]]))) %do% {
          distances_tp <- data[.(tp), on = eval(time)]
          data.table::setorderv(distances_tp, unit)
          distances_tp <- dist_fun_dm(distances_tp[, eval(c(lon, lat)), with = FALSE], dist_comp, sparse, batch, batch_ram_opt, dist_cutoff, dist_round, crs,
            st_distance, dist_which, ncores)
          return(distances_tp)
        }
      }
    }
  } else {
    # Cross-sectional application
    data <- stats::na.omit(data)
    if(verbose) message("Computing distances")
    if(any(class(data) == "sf")) {
      distances <- dist_fun_dm(data, dist_comp, sparse, batch, dist_cutoff, dist_round, crs, st_distance, dist_which, ncores)
    } else {
      if(any(class(data) == "data.table")) {
        distances <- dist_fun_dm(data[, eval(c(lon, lat)), with = FALSE], dist_comp, sparse, batch, batch_ram_opt, dist_cutoff, dist_round, crs, st_distance,
          dist_which, ncores)
      } else {
        distances <- dist_fun_dm(data[, c(lon, lat)], dist_comp, sparse, batch, batch_ram_opt, dist_cutoff, dist_round, crs, st_distance, dist_which, ncores)
      }
    }
  }
  distances <- list(distances = distances, dist_cutoff = dist_cutoff, sparse = sparse, panel = panel)
  if(panel) distances <- c(distances, balanced = balanced)
  class(distances) <- "conley_dist"
  return(distances)
}

# Function computing and adjusting distances
dist_fun_dm <- function(distances, dist_comp, sparse, batch, batch_ram_opt, dist_cutoff, dist_round, crs, st_distance, dist_which, ncores) {
  # Compute distances via sf
  if(st_distance) {
    if(!any(class(distances) == "sf")) distances <- sf::st_as_sf(distances, coords = names(distances), crs = crs)
    # Compute distance matrix
    if(is.null(dist_which)) {
      distances <- sf::st_distance(distances) / 1000
    } else {
      distances <- sf::st_distance(distances, which = dist_which) / 1000
    }
    units(distances) <- NULL
    if(sparse) {
      # Adjust distances according to specified cutoff
      distances[distances == 0] <- NaN
      if(!is.null(dist_cutoff)) {
        distances[distances > dist_cutoff] <- 0
        distances <- Matrix::Matrix(distances, sparse = T)
      }
    }
  } else {
    haversine <- (dist_comp == "spherical")
    # Compute distances via haversine distance function and adjust distances according to specified cutoff
    if(dist_comp == "spherical") distances <- (as.matrix(distances) * 0.01745329252) else distances <- (as.matrix(distances) / 1000)
    # Compute distance matrix
    if(is.null(dist_cutoff)) {
      distances <- dist_mat_d(distances, NROW(distances), haversine, ncores)
    } else {
      if(sparse) {
        if(batch) {
          if(dist_round) {
            if(ncores > 1) {
              distances <- dist_spmat_d_d_b_r_p(distances, NROW(distances), dist_cutoff, haversine, batch_ram_opt, ncores)
            } else {
              distances <- dist_spmat_d_d_b_r(distances, NROW(distances), dist_cutoff, haversine, batch_ram_opt)
            }
          } else {
            if(ncores > 1) {
              distances <- dist_spmat_d_d_b_p(distances, NROW(distances), dist_cutoff, haversine, batch_ram_opt, ncores)
            } else {
              distances <- dist_spmat_d_d_b(distances, NROW(distances), dist_cutoff, haversine, batch_ram_opt)
            }
          }
        } else {
          if(dist_round) {
            distances <- dist_spmat_d_d_r(distances, NROW(distances), dist_cutoff, haversine, ncores)
          } else {
            distances <- dist_spmat_d_d(distances, NROW(distances), dist_cutoff, haversine, ncores)
          }
        }
      } else {
        distances <- dist_mat_d_d(distances, NROW(distances), dist_cutoff, haversine, ncores)
      }
    }
  }
  return(distances)
}

