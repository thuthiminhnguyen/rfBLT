# Apply the Taken Theorem for predictions based on rate of change and past values
taken_theorem_predict <- function(rate_of_change, past_values, time_step = 1) {
  n <- length(past_values)

  # Ensure dimensions of inputs match
  if (length(as.vector(rate_of_change)) != length(as.vector(past_values))) {
    stop("Dimensions of 'rate_of_change' and 'past_values' must match.")
  }

  # Handle time_step input
  time_step <- if (length(time_step) == 1) {
    rep(time_step, n)
  } else if (length(time_step) == n) {
    time_step
  } else {
    stop("Please check the dimension of input 'time_step'.")
  }

  # Calculate predictions using vectorized operations
  pred <- rate_of_change * time_step + past_values

  return(pred)
}
