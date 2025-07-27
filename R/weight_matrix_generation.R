# Weight Matrix Generation
generate_random_weights_uniform <- function(input_dim, output_dim, min_val=-1, max_val=1) {
  W <- matrix(runif(input_dim*output_dim, min=min_val, max=max_val), nrow=input_dim, ncol=output_dim)
  return(W)
}
generate_random_weights_cauchy <- function(input_dim, output_dim, location=0, scale=1) {
  W <- matrix(rcauchy(input_dim*output_dim, location=location, scale=scale), nrow=input_dim, ncol=output_dim)
  return(W)
}
generate_random_weights_exponential <- function(input_dim, output_dim, rate=1) {
  W <- matrix(rexp(input_dim*output_dim, rate=rate), nrow=input_dim, ncol=output_dim)
  return(W)
}
generate_random_weights_bernoulli <- function(input_dim, output_dim, prob=0.5) {
  W <- matrix(rbinom(input_dim*output_dim, size=1, prob=prob), nrow=input_dim, ncol=output_dim)
  return(W)
}
generate_random_weights_lognormal <- function(input_dim, output_dim, meanlog=0, sdlog=1) {
  W <- matrix(rlnorm(input_dim*output_dim, meanlog=meanlog, sdlog=sdlog), nrow=input_dim, ncol=output_dim)
  return(W)
}
generate_random_weights_normal <- function(input_dim, output_dim, mean=0, sd=1) {
  W <- matrix(rnorm(input_dim*output_dim, mean=mean, sd=sd), nrow=input_dim, ncol=output_dim)
  return(W)
}

# Combination of weight matrix generation, look for distribution in R documentation for more information
generate_random_weights <- function(input_dim, output_dim, weight_dist="normal", weight_params=list()) {

  # Check valid input for parameters of weight matrix generation distribution
  validate_params(dist=weight_dist, params=weight_params, name="weight_params")

  if (weight_dist=="uniform") {
    min_val <- ifelse(!is.null(weight_params$min_val), weight_params$min_val, -1)
    max_val <- ifelse(!is.null(weight_params$max_val), weight_params$max_val, 1)
    W <- generate_random_weights_uniform(input_dim, output_dim, min_val, max_val)

  } else if (weight_dist=="normal") {
    mean <- ifelse(!is.null(weight_params$mean), weight_params$mean, 0)
    sd <- ifelse(!is.null(weight_params$sd), weight_params$sd, 1)
    W <- generate_random_weights_normal(input_dim, output_dim, mean, sd)

  } else if (weight_dist=="cauchy") {
    location <- ifelse(!is.null(weight_params$location), weight_params$location, 0)
    scale <- ifelse(!is.null(weight_params$scale), weight_params$scale, 1)
    W <- generate_random_weights_cauchy(input_dim, output_dim, location, scale)

  } else if (weight_dist=="exponential") {
    rate <- ifelse(!is.null(weight_params$rate), weight_params$rate, 1)
    W <- generate_random_weights_exponential(input_dim, output_dim, rate)

  } else if (weight_dist=="bernoulli") {
    prob <- ifelse(!is.null(weight_params$prob), weight_params$prob, 0.5)
    W <- generate_random_weights_bernoulli(input_dim, output_dim, prob)

  } else if (weight_dist=="lognormal") {
    meanlog <- ifelse(!is.null(weight_params$meanlog), weight_params$meanlog, 0)
    sdlog <- ifelse(!is.null(weight_params$sdlog), weight_params$sdlog, 1)
    W <- generate_random_weights_lognormal(input_dim, output_dim, meanlog, sdlog)

  } else {
    stop("Unknown distribution type. Choose from: uniform, normal, cauchy, exponential, bernoulli, lognormal.")

  }

  return(W)
}
