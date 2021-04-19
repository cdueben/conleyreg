#' Conley standard error estimations
#'
#' This function estimates ols, logit, and probit models with Conley standard errors.
#'
#' @param formula regression equation as formula or character string
#' @param data input data, either in non-spatial data frame format (includes tibbles and data tables) with columns denoting coordinates or in sf format with a spatial
#' points geometry. When using a non-spatial data frame format, the coordinates must be longlat. sf objects can use any projection. Note that the projection can influence
#' the computed distances, which is a general phenomenon in GIS software and not specific to \code{conleyreg}.
#' @param dist_cutoff the distance cutoff in km
#' @param model the applied model. Either \code{ols} (default), \code{logit}, or \code{probit}. \code{logit} and \code{probit} are currently restricted to
#' cross-sectional applications.
#' @param unit the variable identifying the cross-sectional dimension. Only needs to be specified, if data is not cross-sectional. Assumes that units do not change their
#' location over time.
#' @param time the variable identifying the time dimension
#' @param lat the variable specifying the latitude in longlat format
#' @param lon the variable specifying the longitude in longlat format
#' @param kernel the kernel applied within the radius. Either \code{bartlett} (default) or \code{uniform}.
#' @param lag_cutoff the cutoff along the time dimension. Defaults to 0, meaning that standard errors are only adjusted cross-sectionally.
#' @param intercept boolean specifying whether to include an intercept. Defaults to \code{TRUE}. Fixed effects models omit the intercept automatically.
#' @param verbose boolean specifying whether to print messages on intermediate estimation steps. Defaults to \code{TRUE}.
#' @param ncores the number of CPU cores to use in the estimations. Defaults to the machine's number of CPUs. Does not affect cross-sectional applications.
#' @param dist_comp choice between \code{precise} (default) and \code{fast} distance computations when data is longlat. Even when choosing \code{precise}, you can still
#' tweak the performance by setting the library that the \code{sf} package uses in distance computations. \code{sf::sf_use_s2(T)} makes it rely on s2 which should be
#' faster than the alternative choice of GEOS with \code{sf::sf_use_s2(F)}. With \code{precise}, distances are great circle distances, with \code{fast} they are
#' haversine distances. Non-longlat data is not affected by this parameter and always uses Euclidean distances.
#'
#' @details This code is an extension and modification of earlier Conley standard error implementations by (i) Richard Bluhm, (ii) Luis Calderon and Leander Heldring,
#' (iii) Darin Christensen and Thiemo Fetzer, and (iv) Timothy Conley. Results vary across implementations because of different distance functions and buffer shapes.
#'
#' @return Returns a \code{lmtest::coeftest} matrix of coefficient estimates and standard errors.
#'
#' @examples
#' \dontrun{
#' # Generate cross-sectional example data
#' data <- data.frame(y = sample(c(0, 1), 100, replace = T),
#'   x1 = stats::runif(100, -50, 50),
#'   lat = runif(100, -90, 90),
#'   lon = runif(100, -180, 180))
#'
#' # Estimate ols model with Conley standard errors using a 1000 km radius
#' conleyreg(y ~ x1, data, 1000, lat = "lat", lon = "lon")
#'
#' # Estimate same model with an sf object as input
#' conleyreg(y ~ x1, sf::st_as_sf(data, coords = c("lon", "lat"), crs = 4326), 1000)
#'
#' # Estimate same model with an sf object of another projection as input
#' conleyreg(y ~ x1, sf::st_transform(sf::st_as_sf(data, coords = c("lon", "lat"), crs = 4326), crs = "+proj=aeqd"), 1000)
#'
#' # Estimate logit model
#' conleyreg(y ~ x1, data, 1000, "logit", lat = "lat", lon = "lon")
#'
#' # Add variable
#' data$x2 <- sample(1:5, 100, replace = T)
#'
#' # Estimate ols model with fixed effects
#' conleyreg(y ~ x1 | x2, data, 1000, lat = "lat", lon = "lon")
#'
#' # Estimate probit model with fixed effects
#' conleyreg(y ~ x1 | x2, data, 1000, "probit", lat = "lat", lon = "lon")
#'
#' # Add panel variables
#' data$time <- rep(1:10, each = 10)
#' data$unit <- rep(1:10, times = 10)
#'
#' # Estimate ols model using panel data
#' conleyreg(y ~ x1, data, 1000, unit = "unit", time = "time", lat = "lat", lon = "lon")
#' }
#'
#'
#' @importFrom foreach %do%
#' @importFrom foreach %dopar%
#' @importFrom data.table :=
#' @importFrom Rdpack reprompt
#' @importFrom Rcpp evalCpp
#'
#' @references
#' \insertCite{*}{conleyreg}
#' \insertAllCited{}
#'
#' @useDynLib conleyreg, .registration = T
#'
#' @exportPattern "ˆ[[:alpha:]]+"
#'
#' @export
conleyreg <- function(formula, data, dist_cutoff, model = c("ols", "logit", "probit"), unit = NULL, time = NULL, lat = NULL, lon = NULL, kernel = c("bartlett", "uniform"),
  lag_cutoff = 0, intercept = T, verbose = T, ncores = NULL, dist_comp = c("precise", "fast")) {

  # Subset the data to variables that are used in the estimation
  if(any(class(data) == "data.table")) {
    data <- data[, eval(unique(c(all.vars(formula), unit, time, lat, lon))), with = F]
  } else {
    data <- data[, unique(c(all.vars(formula), unit, time, lat, lon))]
  }

  # Check spatial attributes
  if(verbose) message("Checking spatial attributes")
  if(any(class(data) == "sf")) {
    # Check if the CRS is set
    if(is.na(sf::st_crs(data))) stop("CRS not set")

    # Check if the CRS is either longlat or uses meters as units (otherwise convert units to meters)
    if(!sf::st_is_longlat(data)) {
      if(gsub("[+]units=", "", regmatches(raster::crs(data)@projargs, regexpr("[+]units=+\\S", raster::crs(data)@projargs))) != "m") {
        data <- sf::st_transform(data, crs = gsub("[+]units=+\\S", "+units=m", raster::crs(data)@projargs))
      }
      longlat <- F
    } else if(sf::st_is_longlat(data)) {
      longlat <- T
    }
  } else if(any(class(data) == "data.frame")) {
    # Check if lat and lon are set
    if(is.null(lat) | is.null(lon)) stop("When data providing data in non-spatial format, you need to specify lat and lon")

    # Check if coordinates are non-missing and longitudes between -180 and 180 and latitudes are between -90 and 90
    if(any(class(data) == "data.table")) {
      ymin <- min(data[[lat]])
      ymax <- max(data[[lat]])
      xmin <- min(data[[lon]])
      xmax <- max(data[[lon]])
    } else {
      ymin <- min(data[, lat, drop = T])
      ymax <- max(data[, lat, drop = T])
      xmin <- min(data[, lon, drop = T])
      xmax <- max(data[, lon, drop = T])
    }
    if(any(is.na(c(ymin, ymax, xmin, xmax)))) stop("Coordinates contain missing values")
    if(any(c(ymin, ymax) < -90) | any(c(ymin, ymax) > 90) | any(c(xmin, xmax) < -180) | any(c(xmin, xmax) > 180)) {
      stop("Coordinates exceed the [-180, 180] interval for longitudes or the [-90, 90] interval for latitudes")
    }
    longlat <- T
  } else {
    stop(paste0("Data of class ", class(data), " is not a valid input"))
  }

  # Check if formula omits intercept
  if(any(grepl("(^|[+]|-|\\s)(0|1)([+]|-|\\s|$)", formula))) stop("Omit the intercept via the intercept argument, not by adding + 0 or - 1 to the formula")

  # Check whether data is panel
  panel <- !is.null(time)

  # Check if panel is balanced
  if(panel) {
    if(is.null(unit)) stop("Cross-sectional identifier, unit, not set")
    if(length(unique(data[[time]])) > 1) {
      balanced <- isbalancedcpp(as.matrix(data.table::setorderv(data.table::data.table(data)[, eval(c(time, unit)), with = F], c(time, unit))))
      if(balanced == 1) balanced <- T
      if(balanced == 0) balanced <- F
      if(balanced == 2) stop(paste0(unit), " does not uniquely identify cross-sectional units")
      if(!balanced & verbose) message("Unbalanced panel identified")
      # Obtain number of observations
      n_obs <- NROW(data)
      # Drop missing values
      data <- stats::na.omit(data)
      if(NROW(data) < n_obs & balanced) {
        balanced <- F
        warning("Panel treated as unbalanced because of missing values")
      }
      if(balanced & verbose) message("Balanced panel identified")
    } else {
      panel <- F
      warning("Number of time periods: 1. Treating data as cross-sectional.")
      # Drop missing values in cross-sectional case
      data <- stats::na.omit(data)
    }
  } else {
    # Drop missing values in cross-sectional case
    data <- stats::na.omit(data)
  }

  # Set model type (default is ols)
  model <- match.arg(model)

  if(model != "ols" & panel) stop("Logit and probit currently exclusively applicable to cross-sectional data")

  # Check if model uses fixed effects
  fe <- any(grepl("[|]", formula))

  # Adjust intercept
  if(!fe & !intercept) formula <- stats::update(formula, ~ . - 1)

  # Estimate model
  if(verbose) message("Estimating model")
  if(model == "ols") {
    reg <- lfe::felm(stats::formula(formula), data = data, keepCX = T)
  } else if(model %in% c("logit", "probit")) {
    if(fe) {
      reg <- bife::bife(stats::formula(formula), data = data, model = model)
    } else {
      reg <- stats::glm(stats::formula(formula), data = data, family = stats::binomial(link = model), x = T)
    }
  }
  # Check if variables are dropped which impedes standard error correction
  if(any(!is.finite(reg$coefficients))) {
    stop("Variables ", paste0(rownames(reg$coefficients)[!is.finite(reg$coefficients)], collapse = ", "), " dropped. Specify a formula that allows to calulate all ",
      "coefficients. Collinearity issues arising from categorical variables can be addressed by using (demeaning) fixed effects (y ~ x1 | x2 + x3) rather than factors ",
      "(y ~ x1 + x2 + x3).")
  }

  # Extract crs and name of geometry column, if data is sf
  if(any(class(data) == "sf")) {
    crs <- sf::st_crs(data)
    sf_col <- attributes(data)$sf_column
  }

  # Extract results
  # Extract coefficients and degrees of freedom
  outp <- reg[c("coefficients", "df.residual")]
  if(model == "ols") {
    # Extract independent variable names
    x_vars <- rownames(reg$coefficients)
    # Extract data
    reg <- data.table::data.table(reg$cX, res = reg$residuals)
    data.table::setnames(reg, (utils::tail(names(reg), 1)), "res")
    if(panel) reg[, eval(c(unit, time)) := data[, (c(unit, time))]]
  } else if(model %in% c("logit", "probit")) {
    # Extract independent variable names
    x_vars <- names(reg$coefficients)
    reg_vcov <- stats::vcov(reg)
    if(fe) {
      reg <- data.table::data.table(reg$data[, eval(x_vars), with = F], res = (reg$data[, 1] - stats::fitted(reg)))
      data.table::setnames(reg, (utils::tail(names(reg), 1)), "res")
    } else {
      reg <- data.table::data.table(reg$x, res = (reg$y - reg$fitted.values))
    }
  }
  if(exists("sf_col")) {
    reg[, eval(sf_col) := data[, (sf_col)]]
  } else {
    if(any(class(data) == "data.table")) reg[, eval(c(lat, lon)) := data[, (c(lat, lon)), with = F]] else reg[, eval(c(lat, lon)) := data[, (c(lat, lon))]]
  }

  # Removes data object as required data was copied to reg
  rm(data)

  # Set kernel (default is bartlett)
  kernel <- match.arg(kernel)

  # Set distance computation type (precise is default)
  dist_comp <- match.arg(dist_comp)

  # Estimate distance matrix
  if(verbose) message(paste0("Estimating distance matri", ifelse(panel, ifelse(balanced, "x", "ces"), "x"), " and addressing spatial correlation"))

  # Set geometry to sf column and respective crs, or to separate coordinate columns if input data was not sf
  if(exists("sf_col")) coords <- list(sf_col, crs) else coords <- c(lon, lat)

  # Obtain number of independent variables
  n_vars <- length(x_vars)

  if(panel) {
    # Panel application
    if(balanced) {
      # Order data (speeds up subsequent computations)
      data.table::setorderv(reg, c(time, unit))
      # Obtain the number of observations per time period
      pl <- NROW(reg) / length(unique(reg[[time]]))
      # Distance matrices are identical across time periods in a balanced panel
      if(is.list(coords)) {
        distances <- dist_fun(reg[1:eval(pl), eval(coords[[1]]), with = F], coords, kernel, dist_cutoff, dist_comp, longlat)
      } else {
        distances <- dist_fun(reg[1:eval(pl), eval(coords), with = F], coords, kernel, dist_cutoff, dist_comp, longlat)
      }
      if(is.null(ncores)) ncores <- parallel::detectCores()
      if(!is.numeric(ncores)) stop("ncores must be either NULL or numeric")
      if(ncores > 1) {
        # Parallel computation
        cl <- parallel::makePSOCKcluster(ncores)
        doParallel::registerDoParallel(cl)
        XeeX <- foreach::foreach(tp = seq(1, NROW(reg), by = pl)) %dopar% {
          reg_tp <- reg[eval(tp):eval((tp + pl - 1)), -eval(unit), with = F]
          return(XeeXhC(distances, as.matrix(reg_tp[, eval(x_vars), with = FALSE]), reg_tp[["res"]], NROW(reg_tp), n_vars))
        }
        parallel::stopCluster(cl)
      } else {
        # Non-parallel computation
        XeeX <- foreach::foreach(tp = seq(1, NROW(reg), by = pl)) %do% {
          reg_tp <- reg[eval(tp):eval((tp + pl - 1)), -eval(unit), with = F]
          return(XeeXhC(distances, as.matrix(reg_tp[, eval(x_vars), with = FALSE]), reg_tp[["res"]], NROW(reg_tp), n_vars))
        }
      }
      # Remove distances object
      rm(distances)
    } else {
      # In unbalanced panels, the distance matrix varies across time periods
      data.table::setkeyv(reg, time)
      if(is.null(ncores)) ncores <- parallel::detectCores()
      if(!is.numeric(ncores)) stop("ncores must be either NULL or numeric")
      if(ncores > 1) {
        # Parallel computation
        cl <- parallel::makePSOCKcluster(ncores)
        doParallel::registerDoParallel(cl)
        XeeX <- foreach::foreach(tp = unique(reg[[time]])) %dopar% {
          reg_tp <- reg[.(tp), -eval(unit), with = F, on = eval(time)]
          if(is.list(coords)) {
            distances <- dist_fun(reg_tp[, eval(coords[[1]]), with = F], coords, kernel, dist_cutoff, dist_comp, longlat)
          } else {
            distances <- dist_fun(reg_tp[, eval(coords), with = F], coords, kernel, dist_cutoff, dist_comp, longlat)
          }
          return(XeeXhC(distances, as.matrix(reg_tp[, eval(x_vars), with = FALSE]), reg_tp[["res"]], NROW(reg_tp), n_vars))
        }
        parallel::stopCluster(cl)
      } else {
        # Non-parallel computation
        XeeX <- foreach::foreach(tp = unique(reg[[time]])) %do% {
          reg_tp <- reg[.(tp), -eval(unit), with = F, on = eval(time)]
          if(is.list(coords)) {
            distances <- dist_fun(reg_tp[, eval(coords[[1]]), with = F], coords, kernel, dist_cutoff, dist_comp, longlat)
          } else {
            distances <- dist_fun(reg_tp[, eval(coords), with = F], coords, kernel, dist_cutoff, dist_comp, longlat)
          }
          return(XeeXhC(distances, as.matrix(reg_tp[, eval(x_vars), with = FALSE]), reg_tp[["res"]], NROW(reg_tp), n_vars))
        }
      }
    }
    XeeX <- Reduce("+",  XeeX)
  } else {
    # Cross-sectional application
    if(is.list(coords)) {
      XeeX <- dist_fun(reg[, eval(coords[[1]]), with = F], coords, kernel, dist_cutoff, dist_comp, longlat)
    } else {
      XeeX <- dist_fun(reg[, eval(coords), with = F], coords, kernel, dist_cutoff, dist_comp, longlat)
    }
    if(model == "ols") {
      # Adressing spatial correlation in case of ols
      XeeX <- XeeXhC(XeeX, as.matrix(reg[, eval(x_vars), with = FALSE]), reg[["res"]], NROW(reg), n_vars)
    } else {
      # Sandwich filling in case of logit and probit
      XeeX <- lp_filling(XeeX, as.matrix(reg[, eval(x_vars), with = FALSE]), reg[["res"]], NROW(reg), n_vars)
    }
  }

  # Drop columns
  if(exists("sf_col")) reg[, eval(sf_col) := NULL] else reg[, eval(c(lon, lat)) := NULL]

  # Obtain number of observations
  n_obs <- NROW(reg)

  if(panel) {
    if(lag_cutoff > 0) {
      if(verbose) message("Addressing serial correlation")
      # Set unit variable as key (speeds up subsetting)
      data.table::setkeyv(reg, unit)
      # In balanced panels, the number of observations per unit is constant across all units
      if(balanced) n_obs_u <- NROW(reg[.(reg[1, eval(unit), with = F]), eval(unit), with = F, on = eval(unit)])
      if(ncores > 1) {
        # Parallel computation
        cl <- parallel::makePSOCKcluster(ncores)
        doParallel::registerDoParallel(cl)
        XeeX_serial <- foreach::foreach(u = unique(reg[[unit]])) %dopar% {
          reg_u <- reg[.(u), on = eval(unit)]
          if(!balanced) n_obs_u <- NROW(reg_u)
          return(time_dist(reg_u[[time]], lag_cutoff, as.matrix(reg_u[, eval(x_vars), with = F]), reg_u[["res"]], n_obs_u, n_vars))
        }
        parallel::stopCluster(cl)
      } else {
        # Non-parallel computation
        XeeX_serial <- foreach::foreach(u = unique(reg[[unit]])) %do% {
          reg_u <- reg[.(u), on = eval(unit)]
          if(!balanced) n_obs_u <- NROW(reg_u)
          return(time_dist(reg_u[[time]], lag_cutoff, as.matrix(reg_u[, eval(x_vars), with = F]), reg_u[["res"]], n_obs_u, n_vars))
        }
      }
      XeeX_serial <- Reduce("+", XeeX_serial)
      XeeX <- XeeX + XeeX_serial
      rm(XeeX_serial)
    } else {
      message("Not addressing serial correlation because lag_cutoff is zero")
    }
  }

  # Compute variance-covariance matrix
  if(model == "ols") {
    V_spatial_HAC <- as.matrix(reg[, eval(x_vars), with = F])
    rm(reg)
    V_spatial_HAC <- solve(crossprod(V_spatial_HAC)) * n_obs
    V_spatial_HAC <- V_spatial_HAC %*% (XeeX / n_obs) %*% V_spatial_HAC / n_obs
    V_spatial_HAC <- (V_spatial_HAC + t(V_spatial_HAC)) / 2
  } else {
    V_spatial_HAC <- lp_vcov(reg_vcov, XeeX, n_vars)
  }

  return(lmtest::coeftest(outp, vcov. = V_spatial_HAC))
}

# Function computing and adjusting distances
dist_fun <- function(distances, coords, kernel, dist_cutoff, dist_comp, longlat) {
  # Compute distances via sf
  if(dist_comp == "precise" | !longlat) {
    # Convert data table to sf format
    if(is.list(coords)) {
      # Coordinates in sf format
      distances <- sf::st_as_sf(distances, crs = coords[[2]], sf_column_name = coords[[1]])
    } else {
      # Coordinates as separate columns
      distances <- sf::st_as_sf(distances, coords = coords, crs = 4326)
    }
    # Compute distance matrix
    distances <- sf::st_distance(distances) / 1000
    units(distances) <- NULL
  } else {
    # Compute distances via haversine distance function
    if(is.list(coords)) {
      # Coordinates in sf format
      distances <- sf::st_coordinates(sf::st_as_sf(distances, crs = coords[[2]], sf_column_name = coords[[1]]))
    } else {
      # Coordinates as separate columns
      distances <- distances[, eval(coords), with = F]
    }
    # Compute distance matrix
    distances <- haversine_mat(as.matrix(distances), NROW(distances))
  }

  # Adjust distances according to specified cutoff
  if(kernel == "bartlett") {
    # The bartlett kernel sets distances above the cutoff to zero and those below to a value between zero and one
    distances <- (1 - distances / dist_cutoff) * (distances <= dist_cutoff)
  } else {
    # The uniform kernel sets distances above the cutoff to zero and those below to one (multiplying by one converts the boolean matrix to numeric)
    distances <- (distances <= dist_cutoff) * 1
  }
  return(distances)
}


