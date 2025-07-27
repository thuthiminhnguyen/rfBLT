#' Plot method for Effective Sample Size Plot  objects
#'
#' @param x An object of class \code{"ess_plot"}.
#' @param ... Additional arguments (currently unused).
#'
#' @importFrom graphics plot.default
#'
#' @export
#'
#' @method plot ess_plot
#'
plot.ess_plot <- function(x, ...) {
  graphics::plot.default(
    x$x,
    x$y,
    main = x$main,
    xlab = x$xlab,
    ylab = x$ylab,
    pch = x$pch,
    ...
  )
}

# Define the function to create a custom plot object
effectiveSize.plot <- function(ess) {
  plot_data <- list(
    x = 1:length(ess),
    y = ess,
    main = "ESS vs Parameter Index",
    xlab = "Parameter Index",
    ylab = "ESS",
    pch = 19
  )
  class(plot_data) <- "ess_plot"  # Assign a custom class
  return(plot_data)
}
