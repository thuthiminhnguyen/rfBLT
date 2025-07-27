# Biases Generation
generate_bias_normal <- function(n, mean=0, sd=1) {
  bias <- rnorm(n, mean=mean, sd=sd)
  return(bias)
}
generate_bias_uniform <- function(n, min_val=0, max_val=2*pi) {
  bias <- runif(n, min=min_val, max=max_val)
  return(bias)
}
generate_bias_exponential <- function(n, rate=1) {
  bias <- rexp(n, rate=rate)
  return(bias)
}
generate_bias_bernoulli <- function(n, prob=0.5) {
  bias <- rbinom(n, size=1, prob=prob)  # Bernoulli distribution (0 or 1)
  return(bias)
}
generate_bias_lognormal <- function(n, meanlog=0, sdlog=1) {
  bias <- rlnorm(n, meanlog=meanlog, sdlog=sdlog)
  return(bias)
}
generate_bias_cauchy <- function(n, location=0, scale=1) {
  bias <- rcauchy(n, location=location, scale=scale)
  return(bias)
}

# Combination of biases generation
generate_bias <- function(n, bias_dist = "uniform", bias_params = list()) {

  # Check valid input for parameters of bias generation distribution
  validate_params(dist=bias_dist, params=bias_params, name="bias_params")

  if (bias_dist == "normal") {
    mean <- ifelse(!is.null(bias_params$mean), bias_params$mean, 0)
    sd <- ifelse(!is.null(bias_params$sd), bias_params$sd, 1)
    bias <- generate_bias_normal(n, mean = mean, sd = sd)

  } else if (bias_dist == "uniform") {
    min_val <- ifelse(!is.null(bias_params$min_val), bias_params$min_val, 0)
    max_val <- ifelse(!is.null(bias_params$max_val), bias_params$max_val, 2*pi)
    bias <- generate_bias_uniform(n, min_val = min_val, max_val = max_val)

  } else if (bias_dist == "exponential") {
    rate <- ifelse(!is.null(bias_params$rate), bias_params$rate, 1)
    bias <- generate_bias_exponential(n, rate = rate)

  } else if (bias_dist == "bernoulli") {
    prob <- ifelse(!is.null(bias_params$prob), bias_params$prob, 0.5)
    bias <- generate_bias_bernoulli(n, prob = prob)  # Bernoulli distribution (0 or 1)

  } else if (bias_dist == "lognormal") {
    meanlog <- ifelse(!is.null(bias_params$meanlog), bias_params$meanlog, 0)
    sdlog <- ifelse(!is.null(bias_params$sdlog), bias_params$sdlog, 1)
    bias <- generate_bias_lognormal(n, meanlog = meanlog, sdlog = sdlog)

  } else if (bias_dist == "cauchy"){
    location <- ifelse(!is.null(bias_params$location), bias_params$location, 0)
    scale <- ifelse(!is.null(bias_params$scale), bias_params$scale, 1)
    bias <- generate_bias_cauchy(n, location = location, scale = scale)

  }  else {
    stop("Unknown bias distribution. Choose from: uniform, normal, cauchy, exponential, bernoulli, lognormal.")
  }

  # Return the generated bias vector
  return(bias)
}
