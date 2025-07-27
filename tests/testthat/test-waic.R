# Assume WAIC_cal_fn_loop and WAIC_cal_fn are already defined in your package or sourced
WAIC_cal_fn_loop <- function(y, X, beta, sigma){
  n <- nrow(X)
  S <- nrow(beta)

  mu_mat <- X %*% t(beta)
  y_mat <- matrix(y, nrow = n, ncol = S)
  residual_mat <- y_mat - mu_mat

  log_pointwise_vec <- numeric(n)
  var_log_vec <- numeric(n)

  for (i in 1:n){
    sample_density <- numeric(S)
    sample_log_density <- numeric(S)
    for (s in 1:S){
      sample_density[s] <- dnorm(residual_mat[i, s], mean=0, sd=sigma[s])
      sample_log_density[s] <- dnorm(residual_mat[i, s], mean=0, sd=sigma[s], log=TRUE)
    }
    log_pointwise_vec[i] <- log(mean(sample_density))
    var_log_vec[i] <- var(sample_log_density)
  }

  lppd <- sum(log_pointwise_vec)
  pWAIC <- sum(var_log_vec)
  WAIC <- -2 * lppd + 2 * pWAIC

  return(list(WAIC = WAIC, lppd = lppd, pWAIC = pWAIC))
}

test_that("WAIC vectorized and loop functions match", {
  set.seed(42)
  n <- 100
  p <- 5
  S <- 500

  # Simulate data
  X <- matrix(rnorm(n * p), nrow = n)
  beta_true <- rnorm(p)
  sigma_true <- 1
  y <- X %*% beta_true + rnorm(n, sd = sigma_true)

  # Simulate posterior samples
  beta_samples <- matrix(rnorm(S * p), nrow = S)
  sigma_samples <- abs(rnorm(S, mean = 1, sd = 0.2))

  # Calculate WAIC from both functions
  waic_loop <- WAIC_cal_fn_loop(y, X, beta_samples, sigma_samples)
  waic_vec <- WAIC_cal_fn(y, X, beta_samples, sigma_samples)

  # Check values are numerically equivalent
  expect_equal(waic_loop$WAIC, waic_vec$WAIC, tolerance = 1e-8)
  expect_equal(waic_loop$lppd, waic_vec$lppd, tolerance = 1e-8)
  expect_equal(waic_loop$pWAIC, waic_vec$pWAIC, tolerance = 1e-8)
})
