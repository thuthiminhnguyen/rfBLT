#' Calculate Time-Series Forward Differences
#'
#' @description Calculates the forward differences (rate of change) for a time-series vector.
#'
#' @details The rate of change is computed using the formula
#' \eqn{\frac{x_{t_2} - x_{t_1}}{t_2 - t_1}}, where \eqn{x_{t}} is the value at time \eqn{t}.
#'
#' @param x A numeric vector.
#' @param step A numeric value or vector specifying time steps.
#'   If a vector is provided, its length (after removing any NAs) must equal `length(x) - 1`.
#'   Defaults to `1`.
forward_difference <- function(x, step = 1) {
  x <- as.vector(x)
  n <- length(x)

  # Handle step input
  if (length(na.omit(step)) == 1) {
    step <- rep(step, (n-1))
  } else if (length(na.omit(step)) == (n-1)) {
    step <- na.omit(step)
  } else {
    stop("Length of 'step' must be either 1 or length(x) - 1.")
  }

  # Vectorized forward difference
  diff_result <- c((x[-1] - x[-n]) / step, NA)

  return(diff_result)
}
