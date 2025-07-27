polynomial_smooth <- function(ts_data, polyorder=5){
  time <- 1:length(ts_data)
  xpoly <- lm(ts_data ~ poly(time, polyorder=polyorder))
  xhat <- xpoly$fitted
  return(xhat)
}

low_pass_filtering_smooth <- function(ts_data, npoint=5){ # npoint is the number of order of the moving average
  filtercoeff <- (1/npoint)*rep(1,npoint=npoint)
  xhat <- stats::filter(ts_data, filtercoeff)
  return(as.vector(xhat))
}

# loess regression - fitting localized polynomials
loess_regression_smooth <- function(ts_data, span=0.1){ # span defines how localized is the fit
  time <- 1:length(ts_data)
  xloess <- loess(ts_data ~ time, span=span)
  xhat <- xloess$fitted
  return(xhat)
}

spline_smooth <- function(ts_data, knotnum=5){
  time <- 1:length(ts_data)
  xspline <- smooth.spline(time, ts_data, nknots=knotnum)
  xhat <- xspline$y
  return(xhat)
}

ma <- function(ts_data, window=5){
  xhat <- rollapply(ts_data, width=window, mean, align="right", partial=TRUE)
  return(xhat)
}

#' Smooth Time Series Data
#'
#' The function applies a smoothing technique to a time series using different methods including
#' polynomial smoothing, low-pass filtering, localized polynomial smoothing, or spline smoothing.
#'
#' @param ts_data A numeric vector representing the time series data to be smoothed.
#' @param fn_name A character string indicating the smoothing method to be used. Options include:
#'        "polynomial", "ma", "localized_polynomial", "spline". Default is "ma".
#' @param params A list of additional parameters for the selected smoothing method.
#'        E.g., for "polynomial", use `params$polyorder`; for "ma", use `params$window`;
#'              for "localized_polynomial" use `params$span`; and for "spline" use `params$knotnum`.
#'
#' @return A numeric vector of the smoothed time series data.
#'
#' @importFrom stats dnorm embed lm loess median na.omit predict quantile
#' @importFrom stats rbinom rcauchy rexp rlnorm rnorm runif smooth.spline var
#' @importFrom utils tail
#' @import zoo
#'
#'
#' @examples
#' # Example with moving average
#' \dontrun{
#' y = cumsum(rnorm(1000))
#' smoothed_data <- smoothness_ts(
#'   ts_data = y,
#'   fn_name = "ma",
#'   params = list(window = 5))
#' }

smoothness_ts <- function(ts_data, fn_name="ma", params = list()){
  if (!is.vector(ts_data)){
    stop("Input must be a vector.")
  }

  if (anyNA(ts_data)){
    ts_data <- na.omit(ts_data)
    warning("NAs are omitted before computing smoothness.")
  }

  if (fn_name == "ma"){
    window <- ifelse(!is.null(params$window), params$window, 5)
    xhat <- ma(ts_data=ts_data, window=window)

  } else if (fn_name == "polynomial"){
    polyorder <- ifelse(!is.null(params$polyorder), params$polyorder, 5)
    xhat <- polynomial_smooth(ts_data=ts_data, polyorder=polyorder)

    # } else if (fn_name == "low_pass_filtering"){
    #   npoint <- ifelse(!is.null(params$npoint), params$npoint, 5)
    #   xhat <- low_pass_filtering_smooth(ts_data=ts_data, npoint=npoint)

  } else if (fn_name == "localized_polynomial"){
    span <- ifelse(!is.null(params$span), params$span, 0.1)
    xhat <- loess_regression_smooth(ts_data=ts_data, span=span)

  } else if (fn_name == "spline"){
    knotnum <- ifelse(!is.null(params$knotnum), params$knotnum, 5)
    xhat <- spline_smooth(ts_data=ts_data, knotnum=knotnum)

  } else {
    stop("Invalid smoothing function name. Please choose from 'ma', 'polynomial', 'localized_polynomial', or 'spline'.")
  }

  return(xhat)
}
