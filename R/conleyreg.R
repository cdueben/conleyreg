#' Conley standard error estimations
#'
#' This function estimates ols, logit, probit, and poisson models with Conley standard errors.
#'
#' @param formula regression equation as formula or character string. Avoid interactions and transformations inside the equation. I.e. use
#' \code{y ~ x1 + x1_2, data = dplyr::mutate(data, x1_2 = x1^2)} instead of \code{y ~ x1 + x1^2, data = data)}.
#' @param data input data. Either (i) in non-spatial data frame format (includes tibbles and data tables) with columns denoting coordinates or (ii) in sf format. In
#' case of an sf object, all non-point geometry types are converted to spatial points, based on the feature's centroid. When using a non-spatial data frame format
#' with projected, i.e. non-longlat, coordinates, \code{crs} must be specified. Note that the projection can influence the computed distances, which is a general
#' phenomenon in GIS software and not specific to \code{conleyreg}. The computationally fastest option is to use a data table with coordinates in the crs in which
#' the distances are to be derived (longlat for spherical and projected for planar), and with time and unit set as keys in the panel case. An sf object as input is
#' the slowest option.
#' @param dist_cutoff the distance cutoff in km
#' @param model the applied model. Either \code{"ols"} (default), \code{"logit"}, \code{"probit"} or \code{"poisson"}. \code{"logit"}, \code{"probit"}, and
#' \code{"poisson"} are currently restricted to cross-sectional applications.
#' @param unit the variable identifying the cross-sectional dimension. Only needs to be specified, if data is not cross-sectional. Assumes that units do not change
#' their location over time.
#' @param time the variable identifying the time dimension. Only needs to be specified, if data is not cross-sectional.
#' @param lat the variable specifying the latitude
#' @param lon the variable specifying the longitude
#' @param kernel the kernel applied within the radius. Either \code{"bartlett"} (default) or \code{"uniform"}.
#' @param lag_cutoff the cutoff along the time dimension. Defaults to 0, meaning that standard errors are only adjusted cross-sectionally.
#' @param intercept logical specifying whether to include an intercept. Defaults to \code{TRUE}. Fixed effects models omit the intercept automatically.
#' @param verbose logical specifying whether to print messages on intermediate estimation steps. Defaults to \code{TRUE}.
#' @param ncores the number of CPU cores to use in the estimations. Defaults to the machine's number of CPUs.
#' @param par_dim the dimension along which the function parallelizes in panel applications. Can be set to \code{"cross-section"} (default) or \code{"time"}. When
#' \code{st_distance = TRUE}, this setting only affects the parallelization in the standard error correction regarding serial correlation, with parallelization in
#' the distance computations automatically set to the time dimension.
#' @param dist_comp choice between \code{"spherical"} and \code{"planar"} distance computations. When unspecified, the input data determines the method: longlat uses
#' spherical (Haversine) distances, alternatives (projected data) use planar (Euclidean) distances. When inserting projected data but specifying
#' \code{dist_comp = "spherical"}, the data is transformed to longlat. Combining unprojected data with \code{dist_comp = "planar"} transforms the data to an
#' azimuthal equidistant format centered at the data's centroid.
#' @param crs the coordinate reference system, if the data is projected. Object of class crs or input string to \code{sf::st_crs}. This parameter can be omitted, if
#' the data is in longlat format (EPSG: 4326), i.e. not projected. If the projection does not use meters as units, this function converts to units to meters.
#' @param st_distance logical specifying whether distances should be computed via \code{sf::st_distance} (\code{TRUE}) or via conleyreg's internal, computationally
#' optimized distance functions (\code{FALSE}). The default (\code{FALSE}) produces the same distances as \code{sf::st_distance} does with S2 enabled. I.e. it uses
#' Haversine (great circle) distances for longlat data and Euclidean distances otherwise. Cases in which you might want to set this argument to \code{TRUE} are e.g.
#' when you want enforce the GEOS approach to computing distances or when you are using an peculiar projection, for which the sf package might include further
#' procedures. Cross-sectional parallelization is not available when \code{st_distance = TRUE} and the function automatically switches to parallelization along the
#' time dimension, if the data is a panel and \code{ncores != 1}. Third and fourth dimensions, termed Z and M in sf, are not accounted for in any case. Note that
#' \code{sf::st_distance} is considerably slower than conleyreg's internal distance functions.
#' @param dist_which the type of distance to use when \code{st_distance = TRUE}. If unspecified, the function defaults to great circle distances for longlat data and
#' to Euclidean distances otherwise. See \code{sf::st_distance} for options.
#' @param sparse logical specifying whether to use sparse rather than dense (regular) matrices in distance computations. Defaults to \code{FALSE}. Only has an effect
#' when \code{st_distance = FALSE}. Sparse matrices are more efficient than dense matrices, when the distance matrix has a lot of zeros arising from points located
#' outside the respective \code{dist_cutoff}. It is recommended to keep the default unless the machine is unable to allocate enough memory.
#' @param batch logical specifying whether distances are inserted into a sparse matrix element by element (\code{FALSE}) or all at once as a batch (\code{TRUE}).
#' Defaults to \code{TRUE}. This argument only has an effect when \code{st_distance = FALSE} and \code{sparse = TRUE}. Batch insertion is faster than element-wise
#' insertion, but requires more memory.
#' @param batch_ram_opt the degree to which batch insertion should be optimized for RAM usage. Can be set to one out of the three levels: \code{"none"},
#' \code{"moderate"} (default), and \code{"heavy"}. Higher levels imply lower RAM usage, but also lower speeds.
#' @param float logical specifying whether distance matrices should use the float (\code{TRUE}) rather than the double (\code{FALSE}) data type. Floats are less
#' precise and than doubles and thereby occupy less space than doubles do. They should only be used when the machine's RAM is insufficient for both the dense and
#' the sparse matrix cases, as they affect the precision of distance values. The \code{float} option only has an effect in Bartlett kernel cases because uniform
#' kernel applications store the data in a smaller integer data type.
#' @param rowwise logical specifying whether to store individual rows of the distance matrix only, instead of the full matrix. If \code{TRUE}, the function uses these
#' rows directly in the standard error correction. This option's advantage is that it induces the function to store only N x \code{ncores} cells, instead of the full
#' N x N matrix, lowering RAM requirements. The disadvantage is that the function needs to compute twice as many distance values as in the default case (\code{FALSE}),
#' since the symmetry of the matrix is not utilized. It hence sacrifices speed for lower RAM utilization. This parameter only has an effect in cross-sectional and
#' unbalanced panel applications with \code{st_distance = FALSE} and \code{sparse = FALSE}.
#' @param reg_ram_opt logical specifying whether the regression should be optimized for RAM usage. Defaults to \code{FALSE}. Changing it to \code{TRUE} slows down
#' the function. This argument only affects the baseline estimation, not the standard error correction.
#' @param dist_mat a distance matrix. Pre-computing a distance matrix and passing it to this argument is only more efficient than having \code{conleyreg} derive it,
#' if you execute \code{conleyreg} multiple times with the same input data. In that case, it is recommended to compute the distance matrix via
#' \code{\link[conleyreg]{dist_mat}}, which is optimized for this purpose and also evaluates various other steps that are identical across regressions on the same
#' input data. Generally, you must not modify the input data between deriving the distance matrix and running \code{conleyreg}. That includes dropping observations
#' or changing values of the unit, time, or coordinate variables. In cross-sectional settings, you must not re-order rows either. If you compute distances through a
#' function other than \code{\link[conleyreg]{dist_mat}}, there are a few additional issues to account for. (i) In the panel scenario, you must order observations by
#' time and unit in ascending order. I.e. cells [1, 2] and [2, 1] of the distance matrix must refer to the distance between unit 1 and unit 2, cells [2, 3] and [3, 2]
#' to the distance between unit 2 and unit 3 etc. The unit numbers in this example refer to their rank when they are sorted. (ii) \code{dist_cutoff} does not refer to
#' kilometers, but to the units of the matrix. (iii) While in a balanced panel you only enter one matrix that is applied to all periods, you supply distances as a
#' list of matrices in the unbalanced case. The matrices must be sorted, with the first list element containing the first period's distance matrix etc. (iv) Zeros in
#' sparse matrices are interpreted as values above the distance cutoff and NaN values are interpreted as zeros. (v) The matrices in the list must all be of the same
#' type - all dense or all sparse. (vi) Distance matrices must only contain non-missing, finite numbers (and NaN in the case of sparse matrices).
#' @param dist_mat_conv logical specifying whether to convert the distance matrix to a list, if the panel turns out to be unbalanced because of missing values. This
#' setting is only relevant, if you enter a balanced panel's distance matrix not derived via \code{\link[conleyreg]{dist_mat}}. If \code{TRUE} (the default), the
#' function only drops rows with missing values. If \code{FALSE}, the function maintains the panel's balanced character by dropping units with missing values in at
#' least one period from the entire data set.
#' @param vcov logical specifying whether to return variance-covariance matrix (\code{TRUE}) rather than the default \code{lmtest::coeftest} matrix of coefficient
#' estimates and standard errors (\code{FALSE}).
#' @param gof logical specifying whether to return goodness of fit measures. Defaults to \code{FALSE}. If \code{TRUE}, the function produces a list.
#'
#' @details This code is an extension and modification of earlier Conley standard error implementations by (i) Richard Bluhm, (ii) Luis Calderon and Leander Heldring,
#' (iii) Darin Christensen and Thiemo Fetzer, and (iv) Timothy Conley. Results vary across implementations because of different distance functions and buffer shapes.
#'
#' This function has reasonable defaults. If your machine has insufficent RAM to allocate the default dense matrices, try sparse matrices. If the RAM error persists,
#' try setting a lower \code{dist_cutoff}, use floats, select a uniform kernel, experiment with \code{batch_ram_opt}, \code{reg_ram_opt}, or \code{batch}.
#'
#' Consult the vignette, \code{vignette("conleyreg_introduction", "conleyreg")}, for a more extensive discussion.
#'
#' @return Returns a \code{lmtest::coeftest} matrix of coefficient estimates and standard errors by default. Can be changed to the variance-covariance matrix by
#' specifying \code{vcov = TRUE}. \insertNoCite{*}{conleyreg}
#'
#' @examples
#' \dontrun{
#' # Generate cross-sectional example data
#' data <- rnd_locations(100, output_type = "data.frame")
#' data$y <- sample(c(0, 1), 100, replace = TRUE)
#' data$x1 <- stats::runif(100, -50, 50)
#'
#' # Estimate ols model with Conley standard errors using a 1000 km radius
#' conleyreg(y ~ x1, data, 1000, lat = "lat", lon = "lon")
#'
#' # Estimate logit model
#' conleyreg(y ~ x1, data, 1000, "logit", lat = "lat", lon = "lon")
#'
#' # Estimate ols model with fixed effects
#' data$x2 <- sample(1:5, 100, replace = TRUE)
#' conleyreg(y ~ x1 | x2, data, 1000, lat = "lat", lon = "lon")
#'
#' # Estimate ols model using panel data
#' data$time <- rep(1:10, each = 10)
#' data$unit <- rep(1:10, times = 10)
#' conleyreg(y ~ x1, data, 1000, unit = "unit", time = "time", lat = "lat", lon = "lon")
#'
#' # Estimate same model with an sf object of another projection as input
#' data <- sf::st_as_sf(data, coords = c("lon", "lat"), crs = 4326) |>
#'   sf::st_transform(crs = "+proj=aeqd")
#' conleyreg(y ~ x1, data, 1000)
#' }
#'
#' @references
#' \insertAllCited{}
#'
#' @importFrom foreach %do%
#' @importFrom foreach %dopar%
#' @importFrom data.table :=
#' @importFrom data.table .N
#' @importFrom data.table .SD
#' @importFrom Rdpack reprompt
#' @importFrom Rcpp evalCpp
#' @importClassesFrom Matrix "Matrix"
#'
#' @useDynLib conleyreg, .registration = TRUE
#'
#' @export
conleyreg <- function(formula, data, dist_cutoff, model = c("ols", "logit", "probit", "poisson"), unit = NULL, time = NULL, lat = NULL, lon = NULL,
  kernel = c("bartlett", "uniform"), lag_cutoff = 0, intercept = TRUE, verbose = TRUE, ncores = NULL, par_dim = c("cross-section", "time"), dist_comp = NULL,
  crs = NULL, st_distance = FALSE, dist_which = NULL, sparse = FALSE, batch = TRUE, batch_ram_opt = NULL, float = FALSE, rowwise = FALSE, reg_ram_opt = FALSE,
  dist_mat = NULL, dist_mat_conv = TRUE, vcov = FALSE, gof = FALSE) {
  # Convert estimation equation to formula, if it was entered as a character string
  formula <- stats::formula(formula)

  # Check whether data is panel
  panel <- !is.null(time)

  # Check whether data is sf object
  sf_data <- any(class(data) == "sf")

  # Check whether dist_mat is null
  null_dist_mat <- is.null(dist_mat)

  # Set model type (default is ols)
  model <- match.arg(model)
  if(model != "ols" && panel) stop("Logit, probit, and poisson currently exclusively applicable to cross-sectional data")

  # Set parallel configuration
  if(is.null(ncores)) {
    ncores <- parallel::detectCores()
    par_dim <- match.arg(par_dim)
  } else if(is.numeric(ncores) && ncores > 0) {
    ncores <- as.integer(ncores)
    par_dim <- match.arg(par_dim)
  } else {
    stop("ncores must be either NULL or a positive integer")
  }

  # Check arguments
  if(any(grepl("(^|[+]|-|\\s)(0|1)([+]|-|\\s|$)", formula))) stop("Omit the intercept via the intercept argument, not by adding + 0 or - 1 to the formula")
  if(panel) {
    if(length(unit) != 1 || !(unit %in% names(data))) stop("In panel applications, unit must be set to the name of variable in the data set")
    if(length(time) != 1 || !(time %in% names(data))) stop("In panel applications, time must be set to the name of variable in the data set")
  }
  if(length(dist_cutoff) != 1 || !is.numeric(dist_cutoff) || dist_cutoff < 0) stop("dist_cutoff must be numeric, >= 0, and of length one")
  if(length(lag_cutoff) != 1 || !is.numeric(lag_cutoff) || lag_cutoff < 0) stop("lag_cutoff must be numeric, >= 0, and of length one")
  if(length(intercept) != 1 || !is.logical(intercept) || is.na(intercept)) stop("intercept must be logical and of length one")
  if(length(verbose) != 1 || !is.logical(verbose) || is.na(verbose)) stop("verbose must be logical and of length one")
  if(length(dist_mat_conv) != 1 || !is.logical(dist_mat_conv) || is.na(dist_mat_conv)) stop("dist_mat_conv must be logical and of length one")
  if(length(vcov) != 1 || !is.logical(vcov) || is.na(vcov)) stop("vcov must be logical and of length one")
  if(length(gof) != 1 || !is.logical(gof) || is.na(gof)) stop("gof must be logical and of length one")

  # Add indicator signaling copied data (if data was copied since function start, data table's set functions can be used without changing the input data)
  data_copied <- FALSE

  # If distances are to be computed via this function
  if(null_dist_mat) {
    # Check arguments
    if(length(st_distance) != 1 || !is.logical(st_distance) || is.na(st_distance)) stop("st_distance must be logical and of length one")
    if(!is.null(dist_comp) && !(dist_comp %in% c("spherical", "planar"))) {
      stop(paste0(dist_comp, ' is not a valid dist_comp value. Choose either NULL (unspecified), "spherical", or "planar".'))
    }
    if(length(sparse) != 1 || !is.logical(sparse) || is.na(sparse)) stop("sparse must be logical and of length one")
    if(length(batch) != 1 || !is.logical(batch) || is.na(batch)) stop("batch must be logical and of length one")
    batch_ram_opt <- which(match.arg(batch_ram_opt, c("moderate", "none", "heavy")) == c("none", "moderate", "heavy"))
    if(length(float) != 1 || !is.logical(float) || is.na(float)) stop("float must be logical and of length one")
    if(length(rowwise) != 1 || !is.logical(rowwise) || is.na(rowwise)) stop("rowwise must be logical and of length one")
    if(st_distance && par_dim == "cross-section" && ncores > 1) {
      par_dim <- "time"
      if(verbose) {
        message('Parallelization along the cross-sectional dimension unavailable when st_distance = TRUE. Thus, par_dim changed to "time". Specify ncores = 1 to run ',
          'the code serially instead.')
      }
    }

    # Subset the data to variables that are used in the estimation
    if(any(class(data) == "data.table")) {
      data <- data[, eval(unique(c(all.vars(formula), unit, time, lat, lon))), with = FALSE]
    } else {
      nc_d <- NCOL(data)
      data <- data[, unique(c(all.vars(formula), unit, time, lat, lon))]
      if(NCOL(data) < nc_d) data_copied <- TRUE
      rm(nc_d)
    }

    # Check spatial attributes
    if(verbose) message("Checking spatial attributes")
    if(sf_data) {
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
      if(sf::st_is_longlat(data)) {
        longlat <- TRUE
      } else {
        if(gsub("[+]units=", "", regmatches(sf::st_crs(data)$proj4string, regexpr("[+]units=+\\S", sf::st_crs(data)$proj4string))) != "m") {
          data <- sf::st_transform(data, crs = gsub("[+]units=+\\S", "+units=m", sf::st_crs(data)$proj4string))
          data_copied <- TRUE
        }
        longlat <- FALSE
      }
    } else if(any(class(data) == "data.frame")) {
      # Check if lat and lon are set
      if(!is.character(lat) || !is.character(lon)) stop("When data providing data in non-spatial format, you need to specify lat and lon")

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
      if(!sf_data) {
        if(is.null(crs)) stop('crs must by specified when entering projected data and using dist_comp == "spherical"')
        data <- sf::st_as_sf(data, coords = c(lon, lat), crs = crs)
      }
      crs <- 4326
      data <- sf::st_transform(data, crs = crs)
    } else if(longlat && dist_comp == "planar") {
      if(!sf_data) data <- sf::st_as_sf(data, coords = c(lon, lat), crs = 4326)
      warning("Projecting longlat data to an azimuthal equidistant format centered at the data's centroid. Note that with this projection, only connections ",
        "passing through the centroid result in correct distance estimates. Other distance estimates can be heavily biased. It is recommended to convert the data to ",
        "a projection that is appropriate for the specific application before inserting it into conleyreg, or to use spherical distance computations.")
      crs <- sf::st_coordinates(sf::st_centroid(sf::st_as_sfc(sf::st_bbox(data))))
      crs <- paste0("+proj=aeqd +lat_0=", crs[1, 2], " +lon_0=", crs[1, 1])
      data <- sf::st_transform(data, crs = crs)
    } else if(st_distance && is.null(crs)) {
      if(sf_data) {
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
    dm_dist_cutoff <- NULL
  } else {
    # If distances are provided via dist_mat
    # Check arguments
    if(verbose && (!is.null(lat) || !is.null(lon) || !is.null(dist_comp) || !st_distance || !is.null(dist_which) || sparse || !batch || float)) {
      message("Arguments lat, lon, dist_comp, st_distance, dist_which, sparse, batch, and float are ignored when dist_mat is not NULL")
    }
    # Drop geometry column
    if(sf_data) {
      data <- sf::st_drop_geometry(data)
      data_copied <- TRUE
    }
    # Subset the data to variables that are used in the estimation
    if(any(class(data) == "data.table")) {
      data <- data[, eval(unique(c(all.vars(formula), unit, time))), with = FALSE]
    } else {
      nc_d <- NCOL(data)
      data <- data[, unique(c(all.vars(formula), unit, time))]
      if(NCOL(data) < nc_d) data_copied <- TRUE
      rm(nc_d)
    }
    # Convert to data table
    if(panel) {
      if(any(class(data) == "data.table")) {
        data.table::setkeyv(data, time)
      } else {
        if(data_copied) {
          data.table::setDT(data, key = time)
        } else {
          data <- data.table::data.table(data, key = time)
        }
      }
    }
    # If distance matrices where computed via dist_mat
    if(any(class(dist_mat) == "conley_dist")) {
      if(panel != dist_mat$panel) {
        if(panel && !dist_mat$panel) {
          stop("Declaring the data to be a panel in this function requires also declaring it a panel when computing the distance matrix")
        } else {
          stop("Declaring the data to be cross-sectional in this function requires also declaring it cross-sectional when computing the distance matrix")
        }
      }
      sparse <- dist_mat$sparse
      dm_dist_cutoff <- dist_mat$dist_cutoff
      if(!is.null(dm_dist_cutoff) && dist_cutoff > dm_dist_cutoff) {
        stop("dist_cutoff must not be larger than the dist_cutoff used in computing the distance matrix")
      }
    # If user generated distance matrices via another function
    } else {
      dm_dist_cutoff <- NULL
      if(is.list(dist_mat)) {
        if(!all(sapply(dist_mat, NROW) == data[, .N, by = time][["N"]])) stop("Distance matrices do not match number of observations in respective time periods")
        sparse <- sapply(dist_mat, function(d_m) methods::is(d_m, "sparseMatrix"))
        if(all(sparse)) {
          sparse <- TRUE
        } else if(all(!sparse)) {
          sparse <- FALSE
        } else {
          stop("The distance matrices must all be of the same type - all dense or all sparse")
        }
      } else if(is.matrix(dist_mat)) {
        if(NROW(data) == NROW(dist_mat)) {
          if(panel) {
            stop("Specified time argument suggests panel, but a distance matrix with as many rows as the input data suggests a cross-sectional structure")
          } else {
            if(verbose) message("Data assumed to be cross-section")
          }
        } else if(NROW(data) %% NROW(dist_mat) != 0) {
          stop("dist_mat neither matches cross-sectional (NROW(data) == NROW(dist_mat)) nor balanced panel (NROW(data) %% NROW(dist_mat) != 0) case. Specify ",
            "dist_mat as a list in an unbalanced panel application.")
        }
      } else {
        stop("dist_mat must be either a matrix (cross-sections, balanced panels) or a list of matrices (unbalanced panels)")
      }
    }
  }

  # Check if panel is balanced
  if(panel) {
    n_tp <- data.table::uniqueN(data, by = time)
    if(n_tp > 1) {
      if(any(class(dist_mat) == "conley_dist")) {
        balanced <- dist_mat$balanced
        data <- stats::na.omit(data)
      } else {
        if(null_dist_mat) {
          balanced <- isbalancedcpp(as.matrix(data.table::setorderv(data[, eval(c(time, unit)), with = FALSE], c(time, unit))))
        } else {
          data.table::setorderv(data, c(time, unit))
          balanced <- isbalancedcpp(as.matrix(data[, eval(c(time, unit)), with = FALSE]))
        }
        if(balanced == 1) {
          balanced <- TRUE
        } else if(balanced == 0) {
          balanced <- FALSE
        } else { # balanced == 2
          stop(paste0(unit), " does not uniquely identify cross-sectional units")
        }
        if(!balanced && verbose) message("Unbalanced panel identified")
        # Drop missing values
        # Distance matrix provided
        if(!null_dist_mat) {
          # Unbalanced panel
          if(is.list(dist_mat)) {
            na_vals <- data[, which(!stats::complete.cases(.SD)), by = time]
            if(NROW(na_vals) > 0) {
              data.table::setkeyv(na_vals, time)
              time_periods <- data.table::data.table(tp_name = unique(data[[time]]), key = "tp_name")[, tp_id := 1:.N]
              for(tp in unique(na_vals[[time]])) {
                tp_id_x <- time_periods[.(tp), tp_id, on = "tp_name"]
                na_vals_x <- na_vals[.(tp), 2, on = time][[1]]
                dist_mat[[tp_id_x]] <- dist_mat[[tp_id_x]][-na_vals_x, -na_vals_x]
              }
              rm(time_periods, tp_id_x, na_vals_x)
            }
          # Balanced panel
          } else {
            na_vals <- which(!stats::complete.cases(data))
            if(length(na_vals) > 0) {
              n_obs_tp <- NROW(data) / n_tp
              # Generate unbalanced panel
              if(dist_mat_conv) {
                if(verbose) message("Missing values detected. Treating panel as unbalanced as dist_mat_conv = TRUE.")
                dist_mat <- lapply(0:(n_tp - 1), function(tp) {
                  na_vals_x <- na_vals[na_vals > (tp * n_obs_tp) & na_vals <= ((tp + 1) * n_obs_tp)]
                  return(dist_mat[-na_vals_x, -na_vals_x])
                })
                balanced <- FALSE
              # Keep balanced panel
              } else {
                if(verbose) message("Missing values detected. Removing affected units from all time periods as dist_mat_conv = FALSE.")
                na_vals <- unique(na_vals %% n_obs_tp)
                dist_mat <- dist_mat[-na_vals, -na_vals]
                na_vals <- sequence(rep(n_tp, times = length(na_vals)), by = n_obs_tp, from = na_vals)
              }
              data <- data[-na_vals,]
              rm(n_obs_tp)
            }
          }
          rm(na_vals)
        # Distance matrix not provided
        } else {
          n_obs <- NROW(data)
          data <- stats::na.omit(data)
          if(NROW(data) < n_obs && balanced) {
            balanced <- FALSE
            warning("Panel treated as unbalanced because of missing values")
          }
          rm(n_obs)
        }
        if(balanced && verbose) message("Balanced panel identified")
      }
    } else {
      panel <- FALSE
      warning("Number of time periods: 1. Treating data as cross-sectional.")
      # Drop missing values in cross-sectional case
      data <- stats::na.omit(data)
    }
    rm(n_tp)
  } else {
    # Drop missing values in cross-sectional case
    data <- stats::na.omit(data)
  }

  # Check if model uses fixed effects
  fe <- any(grepl("[|]", formula))

  # Adjust intercept
  if(!fe && !intercept) formula <- stats::update(formula, ~ . - 1)

  # Drop geometry column of sf object as it may interfere with estimation functions, especially in case of fixest package
  if(any(class(data) == "sf")) {
    sf_col_name <- attributes(data)$sf_column
    sf_col <- data[, sf_col_name]
    data <- sf::st_drop_geometry(data)
  }

  # Estimate model
  if(verbose) message("Estimating model")
  if(model == "ols") {
    reg <- fixest::feols(formula, data = data, nthreads = ncores, demeaned = TRUE, mem.clean = reg_ram_opt)
  } else if(model %in% c("logit", "probit")) {
    reg <- fixest::feglm(formula, data = data, family = stats::binomial(link = model), nthreads = ncores, mem.clean = reg_ram_opt)
  } else { # model == "poisson"
    reg <- fixest::fepois(formula, data = data, nthreads = ncores, mem.clean = reg_ram_opt)
  }
  # Check if variables are dropped which impedes standard error correction
  if(any(!is.finite(reg$coefficients))) {
    stop("Variables ", paste0(rownames(reg$coefficients)[!is.finite(reg$coefficients)], collapse = ", "), " dropped. Specify a formula that allows to calulate all ",
      "coefficients. Collinearity issues arising from categorical variables can be addressed by using (demeaning) fixed effects (y ~ x1 | x2 + x3) rather than factors ",
      "(y ~ x1 + x2 + x3).")
  }

  # Extract results
  # Extract coefficients and degrees of freedom
  if(!vcov) {
    outp <- list()
    outp$coefficients <- reg$coefficients
    outp$df.residual <- reg$nobs - reg$nparams
  }
  res <- "res"
  if(res %in% names(data)) {
    i <- 1
    while(res %in% names(data)) {
      res <- paste0("res", i)
      i <- 1 + 1
    }
    rm(i)
  }
  if(model == "ols") {
    # Extract independent variable names
    x_vars <- names(reg$coefficients)
    if(gof) {
      if(fe) {
        if(length(all.vars(formula)) > 1) gof_outp <- c("r2", "ar2", "wr2", "awr2") else gof_outp <- c("r2", "wr2")
        gof_outp <- c(list(nobs = reg$nobs), fixest::fitstat(reg, gof_outp))
        if(length(gof_outp) == 5) {
          names(gof_outp) <- c("nobs", "r.squared", "adj.r.squared", "within.r.squared", "adj.within.r.squared")
        } else {
          names(gof_outp) <- c("nobs", "r.squared", "within.r.squared")
        }
      } else {
        if(length(all.vars(formula)) > 1) gof_outp <- c("r2", "ar2") else gof_outp <- "r2"
        gof_outp <- c(list(nobs = reg$nobs), fixest::fitstat(reg, gof_outp))
        if(length(gof_outp) == 3) {
          names(gof_outp) <- c("nobs", "r.squared", "adj.r.squared")
        } else {
          names(gof_outp) <- c("nobs", "r.squared")
        }
      }
    }
    # Extract data
    reg <- data.table::data.table(reg$X_demeaned, reg$residuals)
    data.table::setnames(reg, (utils::tail(names(reg), 1)), res)
    if(panel) reg[, eval(c(unit, time)) := data[, eval(c(unit, time)), with = F]]
  } else { # if(model %in% c("logit", "probit", "poisson"))
    # Extract independent variable names
    x_vars <- names(reg$coefficients)
    if(!fe && intercept) {
      if(any(class(data) == "data.table")) {
        data[, "(Intercept)" := 1L]
      } else {
        data$`(Intercept)` <- 1L
      }
    }
    reg_vcov <- reg$cov.unscaled
    if(reg$nobs != reg$nobs_origin) {
      if(any(class(data) == "data.table")) data <- data[-eval(reg$obsRemoved),] else data <- data[-reg$obsRemoved,]
    }
    if(gof) {
      if(length(all.vars(formula)) > 1) gof_outp <- c("pr2", "apr2") else gof_outp <- "pr2"
    }
    if(fe) {
      if(gof) {
        gof_outp <- c(list(nobs = reg$nobs), fixest::fitstat(reg, c(gof_outp, "wpr2")))
        if(length(gof_outp) == 4) {
          names(gof_outp) <- c("nobs", "pseudo.r.squared", "adj.pseudo.r.squared", "within.pseudo.r.squared")
        } else {
          names(gof_outp) <- c("nobs", "pseudo.r.squared", "within.pseudo.r.squared")
        }
      }
      if(any(class(data) == "data.table")) {
        reg <- data.table::data.table(fixest::demean(formula(paste0(paste0(x_vars, collapse = " + "), " ~ ", paste0(reg$fixef_vars, collapse = " + "))),
          data = data.table::setDF(data[, eval(c(x_vars, reg$fixef_vars)), with = FALSE])), reg$residuals)
      } else {
        reg <- data.table::data.table(fixest::demean(formula(paste0(paste0(x_vars, collapse = " + "), " ~ ", paste0(reg$fixef_vars, collapse = " + "))),
          data = data[, c(x_vars, reg$fixef_vars)]), reg$residuals)
      }
      data.table::setnames(reg, (utils::tail(names(reg), 1)), res)
    } else {
      if(gof) {
        gof_outp <- c(list(nobs = reg$nobs), fixest::fitstat(reg, gof_outp))
        if(length(gof_outp) == 3) {
          names(gof_outp) <- c("nobs", "pseudo.r.squared", "adj.pseudo.r.squared")
        } else {
          names(gof_outp) <- c("nobs", "pseudo.r.squared")
        }
      }
      if(any(class(data) == "data.table")) {
        reg <- data[, eval(x_vars), with = FALSE][, eval(res) := reg$residuals]
      } else {
        reg <- data.table::setDT(data[, x_vars])[, eval(res) := reg$residuals]
      }
    }
  }
  if(null_dist_mat) {
    if(exists("sf_col")) {
      reg[, eval(sf_col_name) := sf_col[, sf_col_name]]
    } else {
      if(any(class(data) == "data.table")) {
        reg[, eval(c(lat, lon)) := data[, (c(lat, lon)), with = FALSE]]
      } else {
        reg[, eval(c(lat, lon)) := data[, c(lat, lon)]]
      }
    }
  }

  # Removes data object as required data was copied to reg
  rm(data)

  # Set kernel (default is bartlett)
  kernel <- match.arg(kernel)
  bartlett <- (kernel == "bartlett")

  # Estimate distance matrix
  if(verbose) message(paste0("Estimating distance matri", ifelse(panel, ifelse(balanced, "x", "ces"), "x"), " and addressing spatial correlation"))

  # Obtain number of independent variables
  n_vars <- length(x_vars)

  if(null_dist_mat) {
    # Indicator of whether to use haversine or euclidean distance function
    haversine <- (dist_comp == "spherical")

    # Convert coordinates
    if(!st_distance) {
      if(haversine) {
        reg[, (c(lon, lat)) := lapply(.SD, function(x) x * 0.01745329252), .SDcols = c(lon, lat)]
      } else {
        reg[, (c(lon, lat)) := lapply(.SD, function(x) x / 1000), .SDcols = c(lon, lat)]
      }
    }
  }

  if(panel) {
    # Panel application
    if(balanced) {
      # Order data (speeds up subsequent computations)
      if(null_dist_mat || any(class(dist_mat) == "conley_dist")) data.table::setorderv(reg, c(time, unit))
      # Obtain the number of observations per time period
      pl <- NROW(reg) / data.table::uniqueN(reg, by = time)
      if(null_dist_mat && !st_distance) {
        if(sparse && batch) {
          if(bartlett) {
            if(float) {
              if(ncores > 1) {
                XeeX <- ols_f_b_p(as.matrix(reg[1:eval(pl), eval(c(lon, lat)), with = FALSE]), NROW(reg), pl, dist_cutoff, as.matrix(reg[, eval(x_vars), with = FALSE]),
                  reg[[res]], n_vars, haversine, batch_ram_opt, ncores)
              } else {
                XeeX <- ols_f_b(as.matrix(reg[1:eval(pl), eval(c(lon, lat)), with = FALSE]), NROW(reg), pl, dist_cutoff, as.matrix(reg[, eval(x_vars), with = FALSE]),
                  reg[[res]], n_vars, haversine, batch_ram_opt)
              }
            } else {
              if(ncores > 1) {
                XeeX <- ols_d_b_p(as.matrix(reg[1:eval(pl), eval(c(lon, lat)), with = FALSE]), NROW(reg), pl, dist_cutoff, as.matrix(reg[, eval(x_vars), with = FALSE]),
                  reg[[res]], n_vars, haversine, batch_ram_opt, ncores)
              } else {
                XeeX <- ols_d_b(as.matrix(reg[1:eval(pl), eval(c(lon, lat)), with = FALSE]), NROW(reg), pl, dist_cutoff, as.matrix(reg[, eval(x_vars), with = FALSE]),
                  reg[[res]], n_vars, haversine, batch_ram_opt)
              }
            }
          } else {
            if(ncores > 1) {
              XeeX <- ols_s_b_p(as.matrix(reg[1:eval(pl), eval(c(lon, lat)), with = FALSE]), NROW(reg), pl, dist_cutoff, as.matrix(reg[, eval(x_vars), with = FALSE]),
                reg[[res]], n_vars, haversine, batch_ram_opt, ncores)
            } else {
              XeeX <- ols_s_b(as.matrix(reg[1:eval(pl), eval(c(lon, lat)), with = FALSE]), NROW(reg), pl, dist_cutoff, as.matrix(reg[, eval(x_vars), with = FALSE]),
                reg[[res]], n_vars, haversine, batch_ram_opt)
            }
          }
        } else {
          XeeX <- ols(as.matrix(reg[1:eval(pl), eval(c(lon, lat)), with = FALSE]), NROW(reg), pl, dist_cutoff, as.matrix(reg[, eval(x_vars), with = FALSE]),
            reg[[res]], n_vars, haversine, sparse, bartlett, float, ncores)
        }
      } else {
        if(null_dist_mat && st_distance) {
          if(exists("sf_col")) {
            distances <- dist_fun(sf::st_as_sf(reg[1:eval(pl), eval(sf_col_name), with = FALSE], crs = crs, sf_column_name = sf_col_name), bartlett, dist_cutoff,
              dist_which)
          } else {
            distances <- dist_fun(reg[1:eval(pl), eval(c(lon, lat)), with = FALSE], bartlett, dist_cutoff, dist_which, lat, lon, crs)
          }
        } else {
          if(any(class(dist_mat) == "conley_dist")) {
            distances <- dist_mat$distances
            rm(dist_mat)
            if(bartlett) {
              if(is.null(dm_dist_cutoff)) {
                distances <- (1 - distances / dist_cutoff) * (distances <= dist_cutoff)
              } else {
                if(sparse) {
                  nz_distances <- Matrix::which(distances != 0)
                } else {
                  nz_distances <- which(distances != 0)
                }
                distances[nz_distances] <- (1 - distances[nz_distances] / dist_cutoff) * (distances[nz_distances] <= dist_cutoff)
                rm(nz_distances)
              }
              distances[is.na(distances)] <- 1
            } else {
              if(is.null(dm_dist_cutoff)) {
                distances <- (distances <= dist_cutoff) * 1L
              } else {
                distances <- (distances <= dist_cutoff & distances != 0) * 1L
                distances[is.na(distances)] <- 1L
              }
            }
          } else {
            units(dist_mat) <- NULL
            distances <- dist_mat
            rm(dist_mat)
            if(bartlett) {
              if(sparse) {
                nz_distances <- Matrix::which(distances != 0)
                distances[nz_distances] <- (1 - distances[nz_distances] / dist_cutoff) * (distances[nz_distances] <= dist_cutoff)
                rm(nz_distances)
                distances[is.na(distances)] <- 1
              } else {
                distances <- (1 - distances / dist_cutoff) * (distances <= dist_cutoff)
              }
            } else {
              if(sparse) {
                distances <- (distances <= dist_cutoff & distances != 0) * 1L
                distances[is.na(distances)] <- 1L
              } else {
                distances <- (distances <= dist_cutoff) * 1L
              }
            }
          }
        }
        n_obs <- NROW(reg)
        if(ncores > 1 && par_dim == "time") {
          # Parallelization along time dimension
          cl <- parallel::makePSOCKcluster(ncores)
          doParallel::registerDoParallel(cl)
          XeeX <- foreach::foreach(tp = seq(1, n_obs, by = pl), .combine = "+") %dopar% {
            reg_tp <- reg[eval(tp):eval((tp + pl - 1)), -eval(c(unit, time)), with = FALSE]
            if(!st_distance && sparse) {
              XeeX_tp <- XeeXhC_s_d_R(distances, as.matrix(reg_tp[, eval(x_vars), with = FALSE]), reg_tp[[res]], pl, pl, n_vars, 1L)
            } else {
              if(bartlett) {
                XeeX_tp <- XeeXhC_d_d_R(distances, as.matrix(reg_tp[, eval(x_vars), with = FALSE]), reg_tp[[res]], pl, pl, n_vars, 1L)
              } else {
                XeeX_tp <- XeeXhC_d_s_R(distances, as.matrix(reg_tp[, eval(x_vars), with = FALSE]), reg_tp[[res]], pl, pl, n_vars, 1L)
              }
            }
            return(XeeX_tp)
          }
          parallel::stopCluster(cl)
        } else {
          # Non-parallel computation or parallelization along cross-sectional dimension
          if(!st_distance && sparse) {
            XeeX <- XeeXhC_s_d_R(distances, as.matrix(reg[, eval(x_vars), with = FALSE]), reg[[res]], n_obs, pl, n_vars, ncores)
          } else {
            if(bartlett) {
              XeeX <- XeeXhC_d_d_R(distances, as.matrix(reg[, eval(x_vars), with = FALSE]), reg[[res]], n_obs, pl, n_vars, ncores)
            } else {
              XeeX <- XeeXhC_d_s_R(distances, as.matrix(reg[, eval(x_vars), with = FALSE]), reg[[res]], n_obs, pl, n_vars, ncores)
            }
          }
        }
        rm(distances, n_obs)
      }
    } else {
      # In unbalanced panels, the distance matrix varies across time periods
      data.table::setkeyv(reg, time)
      tps <- sort(unique(reg[[time]]))
      if(ncores > 1 && par_dim == "time") {
        # Parallel computation along time dimension
        cl <- parallel::makePSOCKcluster(ncores)
        doParallel::registerDoParallel(cl)
        XeeX <- foreach::foreach(TP = 1:length(tps), .combine = "+") %dopar% {
          tp <- tps[TP]
          reg_tp <- reg[.(tp), on = eval(time)]
          if(any(class(dist_mat) == "conley_dist")) data.table::setorderv(reg_tp, unit)
          reg_tp[, eval(c(time, unit)) := NULL]
          n_obs_t <- NROW(reg_tp)
          if(null_dist_mat && !st_distance) {
            if(sparse && batch) {
              if(bartlett) {
                if(float) {
                  XeeX_tp <- ols_f_b(as.matrix(reg_tp[, eval(c(lon, lat)), with = FALSE]), n_obs_t, n_obs_t, dist_cutoff,
                    as.matrix(reg_tp[, eval(x_vars), with = FALSE]), reg_tp[[res]], n_vars, haversine, batch_ram_opt)
                } else {
                  XeeX_tp <- ols_d_b(as.matrix(reg_tp[, eval(c(lon, lat)), with = FALSE]), n_obs_t, n_obs_t, dist_cutoff,
                    as.matrix(reg_tp[, eval(x_vars), with = FALSE]), reg_tp[[res]], n_vars, haversine, batch_ram_opt)
                }
              } else {
                XeeX_tp <- ols_s_b(as.matrix(reg_tp[, eval(c(lon, lat)), with = FALSE]), n_obs_t, n_obs_t, dist_cutoff,
                  as.matrix(reg_tp[, eval(x_vars), with = FALSE]), reg_tp[[res]], n_vars, haversine, batch_ram_opt)
              }
            } else {
              if(rowwise) {
                XeeX_tp <- ols_r(as.matrix(reg_tp[, eval(c(lon, lat)), with = FALSE]), n_obs_t, dist_cutoff, as.matrix(reg_tp[, eval(x_vars), with = FALSE]),
                  reg_tp[[res]], n_vars, haversine, bartlett, float, 1L)
              } else {
                XeeX_tp <- ols(as.matrix(reg_tp[, eval(c(lon, lat)), with = FALSE]), n_obs_t, n_obs_t, dist_cutoff, as.matrix(reg_tp[, eval(x_vars), with = FALSE]),
                  reg_tp[[res]], n_vars, haversine, sparse, bartlett, float, 1L)
              }
            }
          } else {
            if(null_dist_mat && st_distance) {
              if(exists("sf_col")) {
                XeeX_tp <- dist_fun(sf::st_as_sf(reg_tp[, eval(sf_col_name), with = FALSE], crs = crs, sf_column_name = sf_col_name), bartlett, dist_cutoff,
                  dist_which)
              } else {
                XeeX_tp <- dist_fun(reg_tp[, eval(c(lon, lat)), with = FALSE], bartlett, dist_cutoff, dist_which, lat, lon, crs)
              }
            } else {
              if(any(class(dist_mat) == "conley_dist")) {
                XeeX_tp <- dist_mat$distances[[TP]]
                if(bartlett) {
                  if(is.null(dm_dist_cutoff)) {
                    XeeX_tp <- (1 - XeeX_tp / dist_cutoff) * (XeeX_tp <= dist_cutoff)
                  } else {
                    if(sparse) {
                      nz_distances <- Matrix::which(XeeX_tp != 0)
                    } else {
                      nz_distances <- which(XeeX_tp != 0)
                    }
                    XeeX_tp[nz_distances] <- (1 - XeeX_tp[nz_distances] / dist_cutoff) * (XeeX_tp[nz_distances] <= dist_cutoff)
                    rm(nz_distances)
                  }
                  XeeX_tp[is.na(XeeX_tp)] <- 1
                } else {
                  if(is.null(dm_dist_cutoff)) {
                    XeeX_tp <- (XeeX_tp <= dist_cutoff) * 1L
                  } else {
                    XeeX_tp <- (XeeX_tp <= dist_cutoff & XeeX_tp != 0) * 1L
                    XeeX_tp[is.na(XeeX_tp)] <- 1L
                  }
                }
              } else {
                XeeX_tp <- dist_mat[[TP]]
                units(XeeX_tp) <- NULL
                if(bartlett) {
                  if(sparse) {
                    nz_distances <- Matrix::which(XeeX_tp != 0)
                    XeeX_tp[nz_distances] <- (1 - XeeX_tp[nz_distances] / dist_cutoff) * (XeeX_tp[nz_distances] <= dist_cutoff)
                    rm(nz_distances)
                    XeeX_tp[is.na(XeeX_tp)] <- 1
                  } else {
                    XeeX_tp <- (1 - XeeX_tp / dist_cutoff) * (XeeX_tp <= dist_cutoff)
                  }
                } else {
                  if(sparse) {
                    XeeX_tp <- (XeeX_tp <= dist_cutoff & XeeX_tp != 0) * 1L
                    XeeX_tp[is.na(XeeX_tp)] <- 1L
                  } else {
                    XeeX_tp <- (XeeX_tp <= dist_cutoff) * 1L
                  }
                }
              }
            }
            if(!st_distance && sparse) {
              XeeX_tp <- XeeXhC_s_d_R(XeeX_tp, as.matrix(reg_tp[, eval(x_vars), with = FALSE]), reg_tp[[res]], n_obs_t, n_obs_t, n_vars, 1L)
            } else {
              if(bartlett) {
                XeeX_tp <- XeeXhC_d_d_R(XeeX_tp, as.matrix(reg_tp[, eval(x_vars), with = FALSE]), reg_tp[[res]], n_obs_t, n_obs_t, n_vars, 1L)
              } else {
                XeeX_tp <- XeeXhC_d_s_R(XeeX_tp, as.matrix(reg_tp[, eval(x_vars), with = FALSE]), reg_tp[[res]], n_obs_t, n_obs_t, n_vars, 1L)
              }
            }
          }
          return(XeeX_tp)
        }
        parallel::stopCluster(cl)
      } else {
        # Non-parallel computation or parallel computation along cross-sectional dimension
        XeeX <- foreach::foreach(TP = 1:length(tps), .combine = "+") %do% {
          tp <- tps[TP]
          reg_tp <- reg[.(tp), on = eval(time)]
          if(any(class(dist_mat) == "conley_dist")) data.table::setorderv(reg_tp, unit)
          reg_tp[, eval(c(time, unit)) := NULL]
          n_obs_t <- NROW(reg_tp)
          if(null_dist_mat && !st_distance) {
            if(sparse && batch) {
              if(bartlett) {
                if(float) {
                  if(ncores > 1) {
                    XeeX_tp <- ols_f_b_p(as.matrix(reg_tp[, eval(c(lon, lat)), with = FALSE]), n_obs_t, n_obs_t, dist_cutoff,
                      as.matrix(reg_tp[, eval(x_vars), with = FALSE]), reg_tp[[res]], n_vars, haversine, batch_ram_opt, ncores)
                  } else {
                    XeeX_tp <- ols_f_b(as.matrix(reg_tp[, eval(c(lon, lat)), with = FALSE]), n_obs_t, n_obs_t, dist_cutoff,
                      as.matrix(reg_tp[, eval(x_vars), with = FALSE]), reg_tp[[res]], n_vars, haversine, batch_ram_opt)
                  }
                } else {
                  if(ncores > 1) {
                    XeeX_tp <- ols_d_b_p(as.matrix(reg_tp[, eval(c(lon, lat)), with = FALSE]), n_obs_t, n_obs_t, dist_cutoff,
                      as.matrix(reg_tp[, eval(x_vars), with = FALSE]), reg_tp[[res]], n_vars, haversine, batch_ram_opt, ncores)
                  } else {
                    XeeX_tp <- ols_d_b(as.matrix(reg_tp[, eval(c(lon, lat)), with = FALSE]), n_obs_t, n_obs_t, dist_cutoff,
                      as.matrix(reg_tp[, eval(x_vars), with = FALSE]), reg_tp[[res]], n_vars, haversine, batch_ram_opt)
                  }
                }
              } else {
                if(ncores > 1) {
                  XeeX_tp <- ols_s_b_p(as.matrix(reg_tp[, eval(c(lon, lat)), with = FALSE]), n_obs_t, n_obs_t, dist_cutoff,
                    as.matrix(reg_tp[, eval(x_vars), with = FALSE]), reg_tp[[res]], n_vars, haversine, batch_ram_opt, ncores)
                } else {
                  XeeX_tp <- ols_s_b(as.matrix(reg_tp[, eval(c(lon, lat)), with = FALSE]), n_obs_t, n_obs_t, dist_cutoff,
                    as.matrix(reg_tp[, eval(x_vars), with = FALSE]), reg_tp[[res]], n_vars, haversine, batch_ram_opt)
                }
              }
            } else {
              if(rowwise) {
                XeeX_tp <- ols_r(as.matrix(reg_tp[, eval(c(lon, lat)), with = FALSE]), n_obs_t, dist_cutoff, as.matrix(reg_tp[, eval(x_vars), with = FALSE]),
                  reg_tp[[res]], n_vars, haversine, bartlett, float, ncores)
              } else {
                XeeX_tp <- ols(as.matrix(reg_tp[, eval(c(lon, lat)), with = FALSE]), n_obs_t, n_obs_t, dist_cutoff, as.matrix(reg_tp[, eval(x_vars), with = FALSE]),
                  reg_tp[[res]], n_vars, haversine, sparse, bartlett, float, ncores)
              }
            }
          } else {
            if(null_dist_mat && st_distance) {
              if(exists("sf_col")) {
                XeeX_tp <- dist_fun(sf::st_as_sf(reg_tp[, eval(sf_col_name), with = FALSE], crs = crs, sf_column_name = sf_col_name), bartlett, dist_cutoff,
                  dist_which)
              } else {
                XeeX_tp <- dist_fun(reg_tp[, eval(c(lon, lat)), with = FALSE], bartlett, dist_cutoff, dist_which, lat, lon, crs)
              }
            } else {
              if(any(class(dist_mat) == "conley_dist")) {
                XeeX_tp <- dist_mat$distances[[TP]]
                if(bartlett) {
                  if(is.null(dm_dist_cutoff)) {
                    XeeX_tp <- (1 - XeeX_tp / dist_cutoff) * (XeeX_tp <= dist_cutoff)
                  } else {
                    if(sparse) {
                      nz_distances <- Matrix::which(XeeX_tp != 0)
                    } else {
                      nz_distances <- which(XeeX_tp != 0)
                    }
                    XeeX_tp[nz_distances] <- (1 - XeeX_tp[nz_distances] / dist_cutoff) * (XeeX_tp[nz_distances] <= dist_cutoff)
                    rm(nz_distances)
                  }
                  XeeX_tp[is.na(XeeX_tp)] <- 1
                } else {
                  if(is.null(dm_dist_cutoff)) {
                    XeeX_tp <- (XeeX_tp <= dist_cutoff) * 1L
                  } else {
                    XeeX_tp <- (XeeX_tp <= dist_cutoff & XeeX_tp != 0) * 1L
                    XeeX_tp[is.na(XeeX_tp)] <- 1L
                  }
                }
              } else {
                XeeX_tp <- dist_mat[[TP]]
                units(XeeX_tp) <- NULL
                if(bartlett) {
                  if(sparse) {
                    nz_distances <- Matrix::which(XeeX_tp != 0)
                    XeeX_tp[nz_distances] <- (1 - XeeX_tp[nz_distances] / dist_cutoff) * (XeeX_tp[nz_distances] <= dist_cutoff)
                    rm(nz_distances)
                    XeeX_tp[is.na(XeeX_tp)] <- 1
                  } else {
                    XeeX_tp <- (1 - XeeX_tp / dist_cutoff) * (XeeX_tp <= dist_cutoff)
                  }
                } else {
                  if(sparse) {
                    XeeX_tp <- (XeeX_tp <= dist_cutoff & XeeX_tp != 0) * 1L
                    XeeX_tp[is.na(XeeX_tp)] <- 1L
                  } else {
                    XeeX_tp <- (XeeX_tp <= dist_cutoff) * 1L
                  }
                }
              }
            }
            if(!st_distance && sparse) {
              XeeX_tp <- XeeXhC_s_d_R(XeeX_tp, as.matrix(reg_tp[, eval(x_vars), with = FALSE]), reg_tp[[res]], n_obs_t, n_obs_t, n_vars, 1L)
            } else {
              if(bartlett) {
                XeeX_tp <- XeeXhC_d_d_R(XeeX_tp, as.matrix(reg_tp[, eval(x_vars), with = FALSE]), reg_tp[[res]], n_obs_t, n_obs_t, n_vars, 1L)
              } else {
                XeeX_tp <- XeeXhC_d_s_R(XeeX_tp, as.matrix(reg_tp[, eval(x_vars), with = FALSE]), reg_tp[[res]], n_obs_t, n_obs_t, n_vars, 1L)
              }
            }
          }
          return(XeeX_tp)
        }
      }
    }
  } else {
    n_obs <- NROW(reg)
    # Cross-sectional application
    if(null_dist_mat && !st_distance) {
      if(model == "ols") {
        if(sparse && batch) {
          if(bartlett) {
            if(float) {
              if(ncores > 1) {
                XeeX <- ols_f_b_p(as.matrix(reg[, eval(c(lon, lat)), with = FALSE]), n_obs, n_obs, dist_cutoff, as.matrix(reg[, eval(x_vars), with = FALSE]),
                  reg[[res]], n_vars, haversine, batch_ram_opt, ncores)
              } else {
                XeeX <- ols_f_b(as.matrix(reg[, eval(c(lon, lat)), with = FALSE]), n_obs, n_obs, dist_cutoff, as.matrix(reg[, eval(x_vars), with = FALSE]),
                  reg[[res]], n_vars, haversine, batch_ram_opt)
              }
            } else {
              if(ncores > 1) {
                XeeX <- ols_d_b_p(as.matrix(reg[, eval(c(lon, lat)), with = FALSE]), n_obs, n_obs, dist_cutoff, as.matrix(reg[, eval(x_vars), with = FALSE]),
                  reg[[res]], n_vars, haversine, batch_ram_opt, ncores)
              } else {
                XeeX <- ols_d_b(as.matrix(reg[, eval(c(lon, lat)), with = FALSE]), n_obs, n_obs, dist_cutoff, as.matrix(reg[, eval(x_vars), with = FALSE]),
                  reg[[res]], n_vars, haversine, batch_ram_opt)
              }
            }
          } else {
            if(ncores > 1) {
              XeeX <- ols_s_b_p(as.matrix(reg[, eval(c(lon, lat)), with = FALSE]), n_obs, n_obs, dist_cutoff, as.matrix(reg[, eval(x_vars), with = FALSE]),
                reg[[res]], n_vars, haversine, batch_ram_opt, ncores)
            } else {
              XeeX <- ols_s_b(as.matrix(reg[, eval(c(lon, lat)), with = FALSE]), n_obs, n_obs, dist_cutoff, as.matrix(reg[, eval(x_vars), with = FALSE]), reg[[res]],
                n_vars, haversine, batch_ram_opt)
            }
          }
        } else {
          if(rowwise) {
            XeeX <- ols_r(as.matrix(reg[, eval(c(lon, lat)), with = FALSE]), n_obs, dist_cutoff, as.matrix(reg[, eval(x_vars), with = FALSE]), reg[[res]], n_vars,
              haversine, bartlett, float, ncores)
          } else {
            XeeX <- ols(as.matrix(reg[, eval(c(lon, lat)), with = FALSE]), n_obs, n_obs, dist_cutoff, as.matrix(reg[, eval(x_vars), with = FALSE]), reg[[res]], n_vars,
              haversine, sparse, bartlett, float, ncores)
          }
        }
      } else {
        if(sparse && batch) {
          if(bartlett) {
            if(float) {
              if(ncores > 1) {
                XeeX <- lp_f_b_p(as.matrix(reg[, eval(c(lon, lat)), with = FALSE]), as.matrix(reg[, eval(x_vars), with = FALSE]), reg[[res]], n_obs, n_vars,
                  dist_cutoff, haversine, batch_ram_opt, ncores)
              } else {
                XeeX <- lp_f_b(as.matrix(reg[, eval(c(lon, lat)), with = FALSE]), as.matrix(reg[, eval(x_vars), with = FALSE]), reg[[res]], n_obs, n_vars,
                  dist_cutoff, haversine, batch_ram_opt)
              }
            } else {
              if(ncores > 1) {
                XeeX <- lp_d_b_p(as.matrix(reg[, eval(c(lon, lat)), with = FALSE]), as.matrix(reg[, eval(x_vars), with = FALSE]), reg[[res]], n_obs, n_vars,
                  dist_cutoff, haversine, batch_ram_opt, ncores)
              } else {
                XeeX <- lp_d_b(as.matrix(reg[, eval(c(lon, lat)), with = FALSE]), as.matrix(reg[, eval(x_vars), with = FALSE]), reg[[res]], n_obs, n_vars,
                  dist_cutoff, haversine, batch_ram_opt)
              }
            }
          } else {
            if(ncores > 1) {
              XeeX <- lp_s_b_p(as.matrix(reg[, eval(c(lon, lat)), with = FALSE]), as.matrix(reg[, eval(x_vars), with = FALSE]), reg[[res]], n_obs, n_vars,
                dist_cutoff, haversine, batch_ram_opt, ncores)
            } else {
              XeeX <- lp_s_b(as.matrix(reg[, eval(c(lon, lat)), with = FALSE]), as.matrix(reg[, eval(x_vars), with = FALSE]), reg[[res]], n_obs, n_vars, dist_cutoff,
                haversine, batch_ram_opt)
            }
          }
        } else {
          if(rowwise) {
            XeeX <- lp_r(as.matrix(reg[, eval(c(lon, lat)), with = FALSE]), as.matrix(reg[, eval(x_vars), with = FALSE]), reg[[res]], n_obs, n_vars, dist_cutoff,
              haversine, bartlett, float, ncores)
          } else {
            XeeX <- lp(as.matrix(reg[, eval(c(lon, lat)), with = FALSE]), as.matrix(reg[, eval(x_vars), with = FALSE]), reg[[res]], n_obs, n_vars, dist_cutoff,
              haversine, sparse, bartlett, float, ncores)
          }
        }
      }
    } else {
      if(null_dist_mat) {
        if(exists("sf_col")) {
          XeeX <- dist_fun(sf::st_as_sf(reg[, eval(sf_col_name), with = FALSE], crs = crs, sf_column_name = sf_col_name), bartlett, dist_cutoff, dist_which)
        } else {
          XeeX <- dist_fun(reg[, eval(c(lon, lat)), with = FALSE], bartlett, dist_cutoff, dist_which, lat, lon, crs)
        }
      } else {
        if(any(class(dist_mat) == "conley_dist")) {
          XeeX <- dist_mat$distances
          rm(dist_mat)
          if(bartlett) {
            if(is.null(dm_dist_cutoff)) {
              XeeX <- (1 - XeeX / dist_cutoff) * (XeeX <= dist_cutoff)
            } else {
              if(sparse) {
                nz_distances <- Matrix::which(XeeX != 0)
              } else {
                nz_distances <- which(XeeX != 0)
              }
              XeeX[nz_distances] <- (1 - XeeX[nz_distances] / dist_cutoff) * (XeeX[nz_distances] <= dist_cutoff)
              rm(nz_distances)
            }
            XeeX[is.na(XeeX)] <- 1
          } else {
            if(is.null(dm_dist_cutoff)) {
              XeeX <- (XeeX <= dist_cutoff) * 1L
            } else {
              XeeX <- (XeeX <= dist_cutoff & XeeX != 0) * 1L
              XeeX[is.na(XeeX)] <- 1L
            }
          }
        } else {
          XeeX <- dist_mat
          rm(dist_mat)
          units(XeeX) <- NULL
          if(bartlett) {
            if(sparse) {
              nz_distances <- Matrix::which(XeeX != 0)
              XeeX[nz_distances] <- (1 - XeeX[nz_distances] / dist_cutoff) * (XeeX[nz_distances] <= dist_cutoff)
              rm(nz_distances)
              XeeX[is.na(XeeX)] <- 1
            } else {
              XeeX <- (1 - XeeX / dist_cutoff) * (XeeX <= dist_cutoff)
            }
          } else {
            if(sparse) {
              XeeX <- (XeeX <= dist_cutoff & XeeX != 0) * 1L
              XeeX[is.na(XeeX)] <- 1L
            } else {
              XeeX <- (XeeX <= dist_cutoff) * 1L
            }
          }
        }
      }
      if(model == "ols") {
        if(!st_distance && sparse) {
          XeeX <- XeeXhC_s_d_R(XeeX, as.matrix(reg[, eval(x_vars), with = FALSE]), reg[[res]], n_obs, n_obs, n_vars, ncores)
        } else {
          if(bartlett) {
            XeeX <- XeeXhC_d_d_R(XeeX, as.matrix(reg[, eval(x_vars), with = FALSE]), reg[[res]], n_obs, n_obs, n_vars, ncores)
          } else {
            XeeX <- XeeXhC_d_s_R(XeeX, as.matrix(reg[, eval(x_vars), with = FALSE]), reg[[res]], n_obs, n_obs, n_vars, ncores)
          }
        }
      } else {
        if(!st_distance && sparse) {
          XeeX <- lp_filling_s_d_R(XeeX, as.matrix(reg[, eval(x_vars), with = FALSE]), reg[[res]], n_obs, n_vars, ncores)
        } else {
          if(bartlett) {
            XeeX <- lp_filling_d_d_R(XeeX, as.matrix(reg[, eval(x_vars), with = FALSE]), reg[[res]], n_obs, n_vars, ncores)
          } else {
            XeeX <- lp_filling_d_s_R(XeeX, as.matrix(reg[, eval(x_vars), with = FALSE]), reg[[res]], n_obs, n_vars, ncores)
          }
        }
      }
    }
  }

  # Drop columns
  if(null_dist_mat) {
    if(exists("sf_col")) reg[, eval(sf_col_name) := NULL] else reg[, eval(c(lon, lat)) := NULL]
  }

  # Obtain number of observations
  n_obs <- NROW(reg)

  if(panel) {
    if(lag_cutoff > 0) {
      if(verbose) message("Addressing serial correlation")
      # Set unit variable as key (speeds up subsetting)
      data.table::setkeyv(reg, unit)
      # In balanced panels, the number of observations per unit is constant across all units
      if(balanced) n_obs_u <- NROW(reg[.(reg[1]), eval(unit), with = FALSE, on = eval(unit)])
      if(ncores > 1 && par_dim == "cross-section") {
        # Parallel computation along cross-sectional dimension
        cl <- parallel::makePSOCKcluster(ncores)
        doParallel::registerDoParallel(cl)
        XeeX_serial <- foreach::foreach(u = unique(reg[[unit]]), .combine = "+") %dopar% {
          reg_u <- reg[.(u), on = eval(unit)]
          if(!balanced) n_obs_u <- NROW(reg_u)
          return(time_dist(reg_u[[time]], lag_cutoff, as.matrix(reg_u[, eval(x_vars), with = FALSE]), reg_u[[res]], n_obs_u, n_vars, 1L))
        }
        parallel::stopCluster(cl)
      } else {
        # Non-parallel computation or parallel computation along time dimension
        XeeX_serial <- foreach::foreach(u = unique(reg[[unit]]), .combine = "+") %do% {
          reg_u <- reg[.(u), on = eval(unit)]
          if(!balanced) n_obs_u <- NROW(reg_u)
          return(time_dist(reg_u[[time]], lag_cutoff, as.matrix(reg_u[, eval(x_vars), with = FALSE]), reg_u[[res]], n_obs_u, n_vars, ncores))
        }
      }
      XeeX <- XeeX + XeeX_serial
      rm(XeeX_serial)
    } else {
      message("Not addressing serial correlation because lag_cutoff is zero")
    }
  }
  # Compute variance-covariance matrix
  if(model == "ols") {
    V_spatial_HAC <- as.matrix(reg[, eval(x_vars), with = FALSE])
    rm(reg)
    V_spatial_HAC <- solve(crossprod(V_spatial_HAC)) * n_obs
    V_spatial_HAC <- V_spatial_HAC %*% (XeeX / n_obs) %*% V_spatial_HAC / n_obs
    V_spatial_HAC <- (V_spatial_HAC + t(V_spatial_HAC)) / 2
  } else {
    V_spatial_HAC <- lp_vcov(reg_vcov, XeeX, n_vars)
  }
  if(!vcov) V_spatial_HAC <- lmtest::coeftest(outp, vcov. = V_spatial_HAC)
  if(gof) {
    if(vcov) V_spatial_HAC <- list(vcov = V_spatial_HAC) else V_spatial_HAC <- list(coefficients = V_spatial_HAC)
    V_spatial_HAC <- c(V_spatial_HAC, gof_outp)
  }
  return(V_spatial_HAC)
}

# Avoid R CMD check note
utils::globalVariables(c(".", "tp", "u", "TP", "tp_id"))

# Function computing distances via sf and transforming them based on the kernel
dist_fun <- function(distances, bartlett, dist_cutoff, dist_which = NULL, lat = NULL, lon = NULL, crs = NULL) {
  if(!any(class(distances) == "sf")) distances <- sf::st_as_sf(distances, coords = c(lon, lat), crs = crs)
  # Compute distance matrix
  if(is.null(dist_which)) {
    distances <- sf::st_distance(distances) / 1000
  } else {
    distances <- sf::st_distance(distances, which = dist_which) / 1000
  }
  units(distances) <- NULL
  # Adjust distances according to specified cutoff
  if(bartlett) {
    # The bartlett kernel sets distances above the cutoff to zero and those below to a value between zero and one
    distances <- (1 - distances / dist_cutoff) * (distances <= dist_cutoff)
  } else {
    # The uniform kernel sets distances above the cutoff to zero and those below to one (multiplying by one converts the boolean matrix to numeric)
    distances <- (distances <= dist_cutoff) * 1L
  }
  return(distances)
}

.onAttach <- function(libname, pkgname) {
  packageStartupMessage('conleyreg 0.1.5 introduces major changes to the function. Read the manual to learn about the updated arguments.')
}
