#' Fit Bayesian regularization random features model
#'
#' Fit the regularization random features model under Bayesian framework.
#' Currently, only Gaussian distribution is supported.
#' The `bayesreg` package is used to estimate the model coefficients via penalized regression (lasso and ridge).
#'
#' @param x A matrix of features.
#' @param y A response vector.
#' @param n_features A positive integer of the number of random features. Defaults to `100`.
#' @param weight_dist A character string of the name of the distribution of the weight matrix.
#'   Defaults to `"normal"`. Other supported options include `"uniform"`, `"cauchy"`, `"exponential"`, `"bernoulli"`, `"lognormal"`.
#' @param weight_params A list of the parameters for the distribution of the weight matrix.
#'   - If `weight_dist = "uniform"`, the list contains:
#'     - `min_val`: Minimum value for the uniform distribution (numeric). Defaults to `-1`.
#'     - `max_val`: Maximum value for the uniform distribution (numeric). Defaults to `1`.
#'   - If `weight_dist = "normal"`, the list contains:
#'     - `mean`: Mean of the normal distribution (numeric). Defaults to `0`.
#'     - `sd`: Standard deviation of the normal distribution (numeric). Defaults to `1`.
#'   - If `weight_dist = "cauchy"`, the list contains:
#'     - `location`: Location parameter of the Cauchy distribution (numeric). Defaults to `0`.
#'     - `scale`: Scale parameter of the Cauchy distribution (numeric). Defaults to `1`.
#'   - If `weight_dist = "exponential"`, the list contains:
#'     - `rate`: Rate parameter of the exponential distribution (numeric). Defaults to `1`.
#'   - If `weight_dist = "bernoulli"`, the list contains:
#'     - `prob`: Probability value of the Bernoulli distribution (numeric). Defaults to `0.5`.
#'   - If `weight_dist = "lognormal"`, the list contains:
#'     - `meanlog`: Mean of the distribution of the log scale (numeric). Defaults to `0`.
#'     - `sdlog`: Standard deviation of the distribution of the log scale (numeric). Defaults to `1`.
#'
#'   Defaults to an empty list `list()` (which is interpreted differently based on `weight_dist`).
#' @param bias_dist A character string of the name of the distribution of the bias vector.
#'   Defaults to `"uniform"`. Other options include `"normal"`, `"exponential"`, `"cauchy"`, `"bernoulli"`, `"lognormal"`.
#' @param bias_params A list of the parameters for the distribution of the bias vector.
#'   - If `bias_dist = "normal"`, the list contains:
#'     - `mean`: Mean of the normal distribution (numeric). Defaults to `0`.
#'     - `sd`: Standard deviation of the normal distribution (numeric). Defaults to `1`.
#'   - If `bias_dist = "uniform"`, the list contains:
#'     - `min_val`: Minimum value for the uniform distribution (numeric). Defaults to `0`.
#'     - `max_val`: Maximum value for the uniform distribution (numeric). Defaults to `2*pi`.
#'   - If `bias_dist = "cauchy"`, the list contains:
#'     - `location`: Location parameter of the Cauchy distribution (numeric). Defaults to `0`.
#'     - `scale`: Scale parameter of the Cauchy distribution (numeric). Defaults to `1`.
#'   - If `bias_dist = "exponential"`, the list contains:
#'     - `rate`: Rate parameter of the exponential distribution (numeric). Defaults to `1`.
#'   - If `bias_dist = "bernoulli"`, the list contains:
#'     - `prob`: Probability value of the Bernoulli distribution (numeric). Defaults to `0.5`.
#'   - If `bias_dist = "lognormal"`, the list contains:
#'     - `meanlog`: Mean of the distribution of the log scale (numeric). Defaults to `0`.
#'     - `sdlog`: Standard deviation of the distribution of the log scale (numeric). Defaults to `1`.
#'
#'   Defaults to a list of parameters of the uniform distribution `list(min_val=0, max_val=2*pi)`.
#' @param act_func A character string of the activation function of the random features model.
#'   Possible values: `"fourier"` (default), `"sigmoid"`, `"tanh"`, `"sine"`, `"cosine"`, `"relu"`.
#' @param reg_type A character string of the regularization method.
#'   Possible values: `"lasso"` (default), `"ridge"`.
#' @param burnin A numeric value specifying the number of burn-in samples. Defaults to `1000`.
#' @param n.samples A numeric value indicating the number of posterior samples to generate. Defaults to `2000`.
#' @param thin A numeric value suggesting the level of thinning. Defaults to `5`.
#' @param n.cores The number of cores for calculation defined by the `bayesreg` package. Defaults to `1`.
#' @param pred_type A character string indicating how to summarize the posterior predictive distribution.
#'   Possible values: `"mean"` (default), `"median"`.
#' @param CI A numeric value indicating the level of the credible interval to report (in percentage). Defaults to `95`.
#'
#' @details
#' The model is defined as:
#' \deqn{\mathbf{y} = \beta_0+\sigma(\mathbf{XW} + \mathbf{b}) \boldsymbol{\beta}+\epsilon}
#' where \eqn{\sigma} is the activation function, \eqn{\mathbf{X}} is the input matrix, \eqn{\mathbf{W}} is the random weighting matrix, \eqn{\mathbf{b}} is the random bias vector,
#' \eqn{\beta_0} is the estimated intercept, \eqn{\boldsymbol{\beta}} is the learnt weights matrix using regularization techniques, and \eqn{\epsilon\sim \mathcal{N}(0, \sigma_{\epsilon}^2)} represents errors.
#'
#' Note that \eqn{\beta_0}, \eqn{\boldsymbol{\beta}}, and \eqn{\sigma_{\epsilon}^2} are estimated using \code{bayesreg::bayesreg}.
#'
#' @return A list of class `"BayesRandFeatRegModel"` containing the fitted model with components:
#'
#' \code{fit_results}: The result of fitted model from `bayesreg::bayesreg`.
#'
#' \code{W}: The matrix of random weights used of random features model.
#'
#' \code{b}: The vector of random biases used of random features model.
#'
#' \code{family}: A character string for the likelihood family.
#'
#' \code{act_func}: A character string of the activation function used in random features model.
#'
#' \code{CI}: The level of the credible interval to report.
#'
#' \code{pred_type}: A character string of the prediction summary type (`"mean"` or `"median"`).
#'
#' \code{train_preds}: A matrix containing the predicted values (`mean` or `median` depending on `pred_type`), along with the lower and upper bounds of the credible interval based on the specified `CI`.
#'
#' \code{posterior_samples}: A matrix of posterior samples of model coefficients.
#'
#' \code{posterior_sigma2}: A numeric vector of posterior samples of the error variance.
#'
#' \code{ess}: An \code{"ess_plot"} object including effective sample sizes (ESS) for MCMC diagnostics.
#'
#' \code{WAIC}: The Watanabe–Akaike information criterion of the fitted model.
#'
#' @examples
#' \dontrun{
#' set.seed(456)
#' x = cbind(rnorm(1000), runif(1000, min = -1, max = 1))
#' y = x[,1] + 2*x[,2] + x[,1]^2 + x[,1]^3
#'
#' fitted <- fit_bayes_reg_rfm(x, y, n_features = 50)
#' }
#'
#' @import bayesreg
#'
#' @export
#'
#' @references
#' `bayesreg`: Bayesian Regression Models with Global-Local Shrinkage Priors
#'
#' CRAN package manual: \url{https://cran.r-project.org/web/packages/bayesreg/index.html}
#'
# Bayesian: https://cran.r-project.org/web/packages/bayesreg/bayesreg.pdf
fit_bayes_reg_rfm <- function(x,
                              y,
                              n_features = 100,
                              weight_dist = "normal",
                              weight_params = list(),
                              bias_dist = "uniform",
                              bias_params = list(min_val = 0, max_val = 2*pi),
                              act_func = "fourier",
                              reg_type = "lasso",
                              # Bayesian
                              burnin = 1000,
                              n.samples = 2000,
                              thin = 5,
                              n.cores = 1,
                              pred_type = "mean",
                              CI = 95) {

  # Check input
  if (is.null(x) || is.null(y)) stop("Please input your data.")

  if (!(positive_int_check(n_features))) stop("`n_features` must be a positive integer.")
  if (!(positive_int_check(burnin))) stop("`burnin` must be a positive integer.")
  if (!(positive_int_check(n.samples))) stop("`n.samples` must be a positive integer.")
  if (!(positive_int_check(thin))) stop("`thin` must be a positive integer.")

  if (!(reg_type %in% c("lasso", "ridge"))) stop("`reg_type` is invalid. Please choose `lasso` or `ridge`.")
  if (!(pred_type %in% c("mean", "median"))) stop("`pred_type` is invalid. Please choose `mean` or `median`.")

  if (CI < 0 && CI > 100) stop("`CI` must be between 0 and 100.")

  # Currently support
  family = "gaussian"

  # Transform training data using random feature model (RFM)
  rf_train <- rfm(
    x = x,
    n_features = n_features,
    weight_dist = weight_dist,
    weight_params = weight_params,
    bias_dist = bias_dist,
    bias_params = bias_params,
    act_func = act_func
  )

  Z_train <- rf_train$Z  # Transformed training data
  W_train <- rf_train$W  # Weights used in the transformation
  b_train <- rf_train$b  # Biases used in the transformation

  df <- data.frame(Z_train, y)
  # n.cores default to Inf in bayesreg::bayesreg
  if (is.null(n.cores)) {
    bayes_reg_model <- bayesreg::bayesreg(
      y ~ .,
      df,
      model = family,
      prior = reg_type,
      n.samples = n.samples,
      burnin = burnin,
      thin = thin
    )
  } else {
    bayes_reg_model <- bayesreg::bayesreg(
      y ~ .,
      df,
      model = family,
      prior = reg_type,
      n.samples = n.samples,
      burnin = burnin,
      thin = thin,
      n.cores = n.cores
    )
  }

  # Extract posterior samples, EXCLUDED THE BURN IN PERIOD
  # Only n.samples are returned, the total number of samples generated is equal to burnin+n.samples*thin.
  # Thinning samples:
  # Ex: 1000 posterior draws can be generated by the following 2 ways
  #     1) Generate 1000 draws after convergence and save all of them
  #     2) Generate 10,000 draws after convergence and save every 10 draws
  # => Produce a higher effective sample size since sum(ACF) will be lower -> higher ESS
  # The reason to thin a sample is to reduce memory requirements
  posterior_beta0 <- t(bayes_reg_model$beta0)          # Intercept
  posterior_beta <- t(as.matrix(bayes_reg_model$beta)) # Coefficients
  posterior_sigma2 <- bayes_reg_model$sigma2           # Noise variance
  posterior_sigma <- sqrt(posterior_sigma2)
  modelWAIC <- bayes_reg_model$waic

  posterior_samples <- cbind(posterior_beta0, posterior_beta) # nrow is n.samples, ncol is number of the number of coefficients
  colnames(posterior_samples) <- c("(Intercept)", paste0("Beta_", 1:ncol(posterior_beta)))

  ess <-  effectiveSize.plot(bayes_reg_model$ess)

  # Noise variance, e_i ~ N(0, sigma_i^2) is a vector of error for each sample
  # --- New code ---
  if (family == "gaussian") {
    posterior_noise <- sapply(posterior_sigma, function(sd) {
      rnorm(nrow(Z_train), mean = 0, sd = sd)
    })
  }
  # --- Old code ---
  # posterior_noise_mat <- matrix(NA, ncol = n.samples, nrow = nrow(Z_train), byrow = TRUE)
  # for (i in 1:ncol(posterior_noise_mat)) {
  #   if (family == "gaussian") {
  #     posterior_noise_mat[, i] = rnorm(nrow(posterior_noise_mat),
  #                                      mean = 0,
  #                                      sd = posterior_sigma[i])
  #   }
  # }

  # Prediction of training data
  pred_train_mat <- cbind(rep(1, nrow(Z_train)), Z_train) %*% t(posterior_samples) + posterior_noise
  pred_train <- if (pred_type == "mean") {
    rowMeans(pred_train_mat)
  } else {
    matrixStats::rowMedians(pred_train_mat)
  }

  alpha.CI <- (100 - CI) / 100
  ci_mat <- credible_interval(x = t(pred_train_mat), alpha = alpha.CI)

  train_preds <- cbind(ci_mat[, 1], pred_train, ci_mat[, 2])
  colnames(train_preds) <- c("lowerCI", "pred", "upper_CI")

  # ---- DIC ----
  # Compute posterior mean for beta
  # beta_posterior_mean <- matrix(colMeans(posterior_samples),
  #                               nrow = 1,
  #                               byrow = TRUE)
  # S <- n.samples # Number of sample excluding burnin period
  #
  # Z_train_1 <- cbind(rep(1, nrow(Z_train)), Z_train) # including column of 1 to account for the intercept

  # Compute log-likelihood at posterior mean of beta
  # if (family == "gaussian") {
  #   log_lik_posterior_mean <- log_likelihood_norm_mat(
  #     y = y,
  #     X = Z_train_1,
  #     beta = beta_posterior_mean,
  #     sigma = mean(posterior_sigma)
  #   )
  #
  #   # Compute log-likelihood for each MCMC sample
  #   log_lik_samples <- sapply(1:S, function(i) {
  #     mat_posterior_samples = matrix(posterior_samples[i, ], nrow = 1, byrow = TRUE)
  #     log_likelihood_norm_mat(
  #       y = y,
  #       X = Z_train_1,
  #       beta = mat_posterior_samples,
  #       sigma = posterior_sigma[i]
  #     )
  #   })
  # }

  # Compute the effective number of parameters (p_DIC)
  # p_DIC <- 2 * (log_lik_posterior_mean - mean(log_lik_samples))

  # Compute DIC
  # DIC <- aic_function(log_lik = log_lik_posterior_mean, k = p_DIC)

  # WAIC_res <- WAIC_cal_fn(y=y, X=Z_train_1, beta=posterior_samples, sigma=posterior_sigma)

  model <- list(
    fit_results = bayes_reg_model,
    W = W_train,
    b = b_train,
    family = family,
    act_func = act_func,
    CI = CI,
    pred_type = pred_type,
    train_preds = train_preds,
    posterior_samples = posterior_samples,
    posterior_sigma2 = posterior_sigma2,
    ess = ess,
    # DIC = DIC,
    WAIC = modelWAIC
    # WAIC = WAIC_res$WAIC
    # modelWAIC = modelWAIC
  )
  class(model) <- "BayesRandFeatRegModel"
  return(model)
}

#' Predict function for `BayesRandFeatRegModel` objects
#'
#' The function aims to predict the response value given a new input newx.
#'
#' @param object A `BayesRandFeatRegModel` object.
#' @param newx New input data (a matrix) for prediction.
#' @param ... Additional arguments (ignored).
#'
#' @return A list containing:
#' \itemize{
#'   \item \code{pred}: A numeric vector of predict values.
#'   \item \code{lower}: A numeric vector of the lower bounds of the credible interval.
#'   \item \code{upper}: A numeric vector of the upper bounds of the credible interval.
#'   \item \code{full_samples}: A matrix of all posterior predictive samples.
#' }
#'
#' @export
#'
predict.BayesRandFeatRegModel <- function(object, newx, ...) {

  if (!inherits(object, "BayesRandFeatRegModel")) {
    stop("Object must be of class 'BayesRandFeatRegModel'")
  }

  W <- object$W
  b <- object$b
  act_func <- attr(object, "act_func") %||% "fourier" # Default activation function
  n.samples <- nrow(object$posterior_samples)
  posterior_sigma <- sqrt(object$posterior_sigma2)

  Z_test <- transformation_and_activation(newx, W, b, object$act_func)

  # Add intercept column
  Z_test_full <- cbind(1, Z_test)

  # Noise variance
  # posterior_noise <- matrix(NA,
  #                           ncol = n.samples,
  #                           nrow = nrow(Z_test),
  #                           byrow = TRUE)

  if (object$family == "gaussian") {
    # --- New code ---
    posterior_noise <- sapply(posterior_sigma, function(sd) {
      rnorm(nrow(Z_test), mean = 0, sd = sd)
    })
    # --- Old code ---
    # for (i in 1:ncol(posterior_noise)) {
    #    posterior_noise[, i] = rnorm(nrow(posterior_noise), mean = 0, sd = posterior_sigma[i])
    # }
  } else {
    stop("Currently, only Gaussian family is supported.")
  }

  preds <- Z_test_full %*% t(object$posterior_samples) + posterior_noise

  alpha <- (100 - object$CI) / 100
  ci <- credible_interval(t(preds), alpha = alpha)
  pred_point <- if (object$pred_type == "mean") rowMeans(preds) else matrixStats::rowMedians(preds)

  return(list(
    pred = pred_point,
    lower = ci[, 1],
    upper = ci[, 2],
    full_samples = preds
  ))
}

#' Forecasting for time series using Bayesian regularization regression with random features model
#'
#' The Bayesian regularization regression with random feature has been applied to predict the values of the time series with the specifying value of time steps.
#' Currently, only Gaussian distribution is supported.
#' The `bayesreg` package is used to estimate the model coefficients via penalized regression.
#'
#' @param ts_data A numeric vector representing the time series data.
#' @param window_size An integer specifying the embedding window size (i.e., the number of past observations used as features). Defaults to `9`.
#' @param pred_size An integer specifying the number of future time steps to predict. Defaults to `7`.
#' @param weight_dist A character string of the name of the distribution of the weight matrix.
#'  Defaults to `"normal"`. Other supported options include `"uniform"`, `"cauchy"`, `"exponential"`, `"bernoulli"`, `"lognormal"`.
#' @param weight_params A list of the parameters for the distribution of the weight matrix.
#'   - If `weight_dist = "uniform"`, the list contains:
#'     - `min_val`: Minimum value for the uniform distribution (numeric). Defaults to `-1`.
#'     - `max_val`: Maximum value for the uniform distribution (numeric). Defaults to `1`.
#'   - If `weight_dist = "normal"`, the list contains:
#'     - `mean`: Mean of the normal distribution (numeric). Defaults to `0`.
#'     - `sd`: Standard deviation of the normal distribution (numeric). Defaults to `1`.
#'   - If `weight_dist = "cauchy"`, the list contains:
#'     - `location`: Location parameter of the Cauchy distribution (numeric). Defaults to `0`.
#'     - `scale`: Scale parameter of the Cauchy distribution (numeric). Defaults to `1`.
#'   - If `weight_dist = "exponential"`, the list contains:
#'     - `rate`: Rate parameter of the exponential distribution (numeric). Defaults to `1`.
#'   - If `weight_dist = "bernoulli"`, the list contains:
#'     - `prob`: Probability value of the Bernoulli distribution (numeric). Defaults to `0.5`.
#'   - If `weight_dist = "lognormal"`, the list contains:
#'     - `meanlog`: Mean of the distribution of the log scale (numeric). Defaults to `0`.
#'     - `sdlog`: Standard deviation of the distribution of the log scale (numeric). Defaults to `1`.
#'
#'   Defaults to an empty list `list()` (which is interpreted differently based on `weight_dist`).
#' @param bias_dist A character string of the name of the distribution of the bias vector.
#'   Defaults to `"uniform"`. Other options include `"normal"`, `"exponential"`, `"cauchy"`, `"bernoulli"`, `"lognormal"`.
#' @param bias_params A list of the parameters for the distribution of the bias vector.
#'   - If `bias_dist = "normal"`, the list contains:
#'     - `mean`: Mean of the normal distribution (numeric). Defaults to `0`.
#'     - `sd`: Standard deviation of the normal distribution (numeric). Defaults to `1`.
#'   - If `bias_dist = "uniform"`, the list contains:
#'     - `min_val`: Minimum value for the uniform distribution (numeric). Defaults to `0`.
#'     - `max_val`: Maximum value for the uniform distribution (numeric). Defaults to `2*pi`.
#'   - If `bias_dist = "cauchy"`, the list contains:
#'     - `location`: Location parameter of the Cauchy distribution (numeric). Defaults to `0`.
#'     - `scale`: Scale parameter of the Cauchy distribution (numeric). Defaults to `1`.
#'   - If `bias_dist = "exponential"`, the list contains:
#'     - `rate`: Rate parameter of the exponential distribution (numeric). Defaults to `1`.
#'   - If `bias_dist = "bernoulli"`, the list contains:
#'     - `prob`: Probability value of the Bernoulli distribution (numeric). Defaults to `0.5`.
#'   - If `bias_dist = "lognormal"`, the list contains:
#'     - `meanlog`: Mean of the distribution of the log scale (numeric). Defaults to `0`.
#'     - `sdlog`: Standard deviation of the distribution of the log scale (numeric). Defaults to `1`.
#'
#'   Defaults to a list of parameters of the uniform distribution `list(min_val=0, max_val=2*pi)`.
#' @param act_func A character string of the activation function of the random features model.
#'   Possible values: `"fourier"` (default), `"sigmoid"`, `"tanh"`, `"sine"`, `"cosine"`, `"relu"`.
#' @param reg_type A character string of the regularization method.
#'   Possible values: `"lasso"` (default), `"ridge"`.
#' @param burnin A numeric value specifying the number of burn-in samples. Defaults to `1000`.
#' @param n.samples A numeric value indicating the number of posterior samples to generate. Defaults to `2000`.
#' @param thin A numeric value suggesting the level of thinning. Defaults to `5`.
#' @param n.cores The number of cores for calculation defined by the `bayesreg` package. Defaults to `1`.
#' @param pred_type A character string indicating how to summarize the posterior predictive distribution.
#'   Possible values: `"mean"` (default), `"median"`.
#' @param feature_selection A character string specifying the method for selecting the number of random features.
#'   Possible values are:
#'   - `"sqrt"` (default): The number of features is set to the square root of the number of rows of the training matrix `nrow`, i.e., `sqrt(nrow)`.
#'   - `"factor"`: The number of features is `feature_constant*nrow`.
#'   - `"constant"`: The number of features is fixed at `feature_constant`.
#'   Note that the number of features will be always rounded down.
#' @param feature_constant A numeric value used when `feature_selection` is `"factor"` or `"constant"`.
#'   Ignored when `feature_selection = "sqrt"`. Defaults to `NULL`.
#' @param CI A numeric value indicating the level of the credible interval to report (in percentage). Defaults to `95`.
#'
#' @details
#' Given a time series \eqn{\{(y_{t_k})\}_{k=1}^n}, the embedding matrix with the window size of \eqn{m} is constructed as follows:
#' \deqn{
#' \mathbf{X} \gets
#' \begin{bmatrix}
#' y_{t_1} & y_{t_2} & \dots & y_{t_{m-1}} & y_{t_m} \\
#' y_{t_2} & y_{t_3} & \dots & y_{t_m} & y_{t_{m+1}} \\
#' \vdots & \vdots & & \vdots & \vdots \\
#' y_{t_{n-m}} & y_{t_{n-m+1}} & \dots & y_{t_{n-2}} & y_{t_{n-1}}
#' \end{bmatrix}
#' }
#' The corresponding output vector is defined as:
#' \deqn{
#' \mathbf{y} \gets
#' \begin{bmatrix}
#' y_{t_{m+1}} \\
#' y_{t_{m+2}} \\
#' \vdots \\
#' y_{t_{n}}
#' \end{bmatrix}
#' }
#' The model is defined as:
#' \deqn{\mathbf{y} = \beta_0+\sigma(\mathbf{XW} + \mathbf{b}) \boldsymbol{\beta}+\epsilon}
#' where \eqn{\sigma} is the activation function, \eqn{\mathbf{X}} is the input matrix, \eqn{\mathbf{W}} is the random weighting matrix, \eqn{\mathbf{b}} is the random bias vector,
#' \eqn{\beta_0} is the estimated intercept, \eqn{\boldsymbol{\beta}} is the learnt weights matrix using regularization techniques, and \eqn{\epsilon\sim \mathcal{N}(0, \sigma_{\epsilon}^2)} represents errors.
#'
#' Note that \eqn{\beta_0}, \eqn{\boldsymbol{\beta}}, and \eqn{\sigma_{\epsilon}^2} are estimated using \code{bayesreg::bayesreg}.
#'
#' @return A list containing the fitted model and relevant components:
#'
#' \code{fit_results}: The result of fitted model from \code{bayesreg::bayesreg}.
#'
#' \code{fitted.values}: The fitted mean or median values defined by `pred_type`.
#'
#' \code{W}: The matrix of random weights used of random features model.
#'
#' \code{b}: The vector of random biases used of random features model.
#'
#' \code{y_pred}: A numeric vector of forecast values for future time steps, with length equal to \code{pred_size}.
#'
#' \code{future_preds}: A matrix of all posterior predictive samples for future time steps.
#'
#' \code{n_features}: A numeric value of the number of random features.
#'
#' \code{posterior_samples}: A matrix of posterior samples of model coefficients.
#'
#' \code{posterior_sigma2}: A numeric vector of posterior samples of the error variance.
#'
#' \code{ess}: An \code{"ess_plot"} object including effective sample sizes (ESS) for MCMC diagnostics.
#'
#' \code{WAIC}: The Watanabe–Akaike information criterion of the fitted model.
#'
#' \code{pred.ci}: A matrix including the credible interval based on the input of \code{CI}.
#'
#' @examples
#' \dontrun{
#' set.seed(456)
#' y = cumsum(rnorm(1000))
#' fitted_bayes_reg_rfm <- ts_forecast_bayes_reg_rfm(ts_data=y)
#' }
#'
#' @import bayesreg
#'
#' @export
#'
#' @references
#' `bayesreg`: Bayesian Regression Models with Global-Local Shrinkage Priors
#'
#' CRAN package manual: \url{https://cran.r-project.org/web/packages/bayesreg/index.html}
#'

# Forecasting for time series
ts_forecast_bayes_reg_rfm <- function(ts_data,
                                      window_size = 9,
                                      pred_size = 7,
                                      weight_dist = "normal",
                                      weight_params = list(),
                                      bias_dist = "uniform",
                                      bias_params = list(min_val = 0, max_val = 2*pi),
                                      act_func = "fourier",
                                      reg_type = "lasso",
                                      burnin = 1000,
                                      n.samples = 2000,
                                      thin = 5,
                                      n.cores = 1,
                                      pred_type = "mean",
                                      feature_selection = "sqrt",
                                      feature_constant = NULL,
                                      CI = 95) {
  # Check input
  if (is.null(ts_data)) stop("Please check your input time series.")
  if (!(positive_int_check(window_size))) stop("`window_size` must be a positive integer.")
  if (!(positive_int_check(pred_size))) stop("`pred_size` must be a positive integer.")
  if (!(pred_type %in% c("mean", "median"))) stop("`pred_type` is invalid. Please choose 'mean' or 'median'.")

  if (window_size > length(ts_data)) stop("`window_size` exceeds `length(ts_data)`.")

  # Currently support
  family = "gaussian"

  train_dat_vec <- as.numeric(ts_data)
  embed_dat_train <- embed(train_dat_vec, dimension = window_size + 1)
  # Embed training data
  x_train <- embed_dat_train[, 2:(window_size + 1)]
  # Reverse data in order of time
  x_train <- x_train[, window_size:1]

  # Target is the first column of the embed data
  y_train <- embed_dat_train[, 1]
  # Reverse data in order of time and make sure it is in matrix form
  x_test <- matrix(rev(embed_dat_train[nrow(embed_dat_train), 1:window_size]), nrow = 1, byrow = TRUE)

  # Determine number of features based on the selected method
  n_features <- check_num_feature(
    feature_selection = feature_selection,
    feature_constant = feature_constant,
    n = nrow(x_train)
  )

  rfm_model <- fit_bayes_reg_rfm(
    x = x_train,
    y = y_train,
    n_features = n_features,
    weight_dist = weight_dist,
    weight_params = weight_params,
    bias_dist = bias_dist,
    bias_params = bias_params,
    act_func = act_func,
    reg_type = reg_type,
    burnin = burnin,
    n.samples = n.samples,
    thin = thin,
    n.cores = n.cores,
    pred_type = pred_type,
    CI = CI
  )

  # Predict on training data
  preds <- predict.BayesRandFeatRegModel(rfm_model, newx = x_test)

  fit_model <- rfm_model$fit_results    # Fitted model
  W <- rfm_model$W                      # Random feature weights
  b <- rfm_model$b                      # Random feature biases
  # DIC <- rfm_model$DIC
  WAIC <- rfm_model$WAIC
  # modelWAIC <- rfm_model$modelWAIC
  train_preds <- rfm_model$train_preds  # Predictions on training set
  posterior_samples <- rfm_model$posterior_samples

  posterior_sigma2 <- rfm_model$posterior_sigma2
  posterior_sigma <- sqrt(posterior_sigma2)

  # Noise variance
  # --- New code ---
  if (family == "gaussian"){
    posterior_noise <- t(sapply(posterior_sigma, function(sd) {
      rnorm((pred_size-1), mean = 0, sd = sd)
    }))
  }
  # --- Old code ---
  # posterior_noise <- matrix(rep(NA, n.samples * (pred_size - 1)), ncol = (pred_size - 1), nrow = n.samples)
  # for (i in 1:n.samples) {
  #   posterior_noise[i, ] <- rnorm((pred_size - 1), mean = 0, sd = posterior_sigma[i])
  # }

  ess <- rfm_model$ess

  y_pred <- rep(NA, pred_size)
  # y_pred[1] <- if (pred_type == "mean") {
  #    mean(preds$full_samples)
  # } else {
  #   median(preds$full_samples)
  # }

  future_preds <- matrix(NA, n.samples, pred_size)
  future_preds[, 1] <- as.vector(preds$full_samples)

  future_preds <- Reduce(function(prev_preds, i) {
    # A matrix will be returned
    x_test <- t(apply(prev_preds, 1, function(row_pred) {
      tail(c(train_dat_vec, row_pred), window_size)
    }))

    # Do transformation
    Z_test <- transformation_and_activation(x_test, W, b, act_func)
    # Add 1 for intercept coefficient
    Z_test_full <- cbind(1, Z_test)

    # Predict next step for each sample
    next_step_preds <- rowSums(Z_test_full * posterior_samples) + posterior_noise[, (i-1)]

    # Append predictions to existing matrix
    cbind(prev_preds, next_step_preds)
  }, 2:pred_size, init = future_preds[, 1, drop = FALSE])

  # Final point predictions
  y_pred <- if (pred_type == "mean") {
    colMeans(future_preds)
  } else {
    matrixStats::colMedians(future_preds)
  }

  alpha.CI = (100 - CI) / 100
  pred_ci_matrix = credible_interval(x = future_preds, alpha = alpha.CI)

  return(
    list(
      fit_results = fit_model,
      fitted.values = train_preds[,2],
      W = W,
      b = b,
      y_pred = as.numeric(y_pred),
      future_preds = future_preds,
      n_features = n_features,
      posterior_samples = posterior_samples,
      posterior_sigma2 = posterior_sigma2,
      ess = ess,
      # DIC = DIC,
      WAIC = WAIC,
      # modelWAIC = modelWAIC,
      pred.ci = pred_ci_matrix
    )
  )
}
