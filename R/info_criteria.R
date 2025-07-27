# --- Log likelikhood of normal distribution ---
# Define the log-likelihood function for a linear regression model for MATRIX
log_likelihood_norm_mat <- function(y, X, beta, sigma) {
  residuals <- y-X%*%t(beta)
  log_lik <- sum(dnorm(residuals, mean=0, sd=sigma, log=TRUE))         # dnorm with log = TRUE
  return(log_lik)
}

# Define the log-likelihood function for a linear regression model for VECTOR
log_likelihood_norm_vec <- function(y, y_pred) {
  residuals <- y-y_pred
  sigma2 <- var(residuals)  # Estimated variance
  log_lik <- sum(dnorm(residuals, mean=0, sd=sqrt(sigma2), log=TRUE))  # dnorm with log = TRUE
  return(log_lik)
}

# WAIC
WAIC_cal_fn <- function(y, X, beta, sigma) {
  n <- nrow(X)
  S <- nrow(beta) # Number of Samples

  mu_mat <- X %*% t(beta)
  y_mat <- matrix(y, nrow = n, ncol = S)
  residual_mat <- y_mat - mu_mat

  log_lik_mat <- matrix(
    mapply(function(res, s) dnorm(res, mean = 0, sd = s, log = TRUE),
           res = as.vector(residual_mat),
           s = rep(sigma, each = n)),
    nrow = n, ncol = S
  )

  log_mean_exp <- function(x_row) {
    max_x <- max(x_row)
    max_x + log(mean(exp(x_row - max_x)))
  }

  log_pointwise_vec <- apply(log_lik_mat, 1, log_mean_exp)
  var_log_vec <- apply(log_lik_mat, 1, var)

  lppd <- sum(log_pointwise_vec)
  pWAIC <- sum(var_log_vec)
  WAIC <- -2 * lppd + 2 * pWAIC

  return(list(WAIC = WAIC, lppd = lppd, pWAIC = pWAIC))
}

# --- AIC ---
aic_function <- function(log_lik, k){
  return(-2*log_lik+2*k)
}
