# Forecast time series data using Takens' theorem under the Bayesian Regularization approach
ts_forecast_bayes_taken_reg_rfm_no_smooth <- function(ts_data,
                                                      delta_t,
                                                      rate_of_change,
                                                      window_size = 9,
                                                      pred_size = 7,
                                                      pred_time_step = 1,
                                                      weight_dist = "normal",
                                                      weight_params = list(),
                                                      bias_dist = "uniform",
                                                      bias_params = list(min_val = 0, max_val = 2*pi),
                                                      act_func = "fourier",
                                                      reg_type = "lasso",
                                                      family = "gaussian",
                                                      # Bayes model
                                                      burnin = 1000,
                                                      n.samples = 2000,
                                                      thin = 5,
                                                      n.cores = 1,
                                                      pred_type = "mean",
                                                      feature_selection = "sqrt",
                                                      feature_constant = NULL,
                                                      CI = 95) {

  train_dat_vec <- ts_data
  embed_dat_train <- embed(train_dat_vec, dimension = window_size + 1)
  if (nrow(embed_dat_train) > length(rate_of_change[window_size:length(rate_of_change)])){
    stop("Please check the size of the input rate of change.")
  }
  x_train <- embed_dat_train[, 2:(window_size + 1)]
  # Reverse the data in time order
  x_train <- x_train[, window_size:1]
  y_train <- rate_of_change[window_size:(nrow(x_train) + window_size - 1)] # consider appropriate index
  delta_t_train <- delta_t[window_size:(nrow(x_train) + window_size - 1)]  # appropriate index of training delta t

  if (nrow(x_train) != length(y_train)){
    stop("Please check dimensions of x_train and y_train.")
  }
  x_test <- matrix(rev(embed_dat_train[nrow(embed_dat_train), 1:window_size]), nrow = 1, byrow = TRUE) # reverse the data

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
  preds <- predict.BayesRandFeatRegModel(rfm_model, newx = x_test, CI = CI)

  fit_model <- rfm_model$fit_results                # Fitted model
  W <- rfm_model$W                                  # Random feature weights
  b <- rfm_model$b                                  # Random feature biases
  # DIC <- rfm_model$DIC
  WAIC <- rfm_model$WAIC
  # modelWAIC <- rfm_model$modelWAIC
  x_derivative_train_preds <- rfm_model$train_preds # Predictions on training set
  fitted.values <- x_derivative_train_preds[,2] * delta_t_train + as.vector(x_train[, window_size])

  posterior_samples <- rfm_model$posterior_samples  # Posterior of coefficients
  posterior_sigma2 <- rfm_model$posterior_sigma2    # Posterior of variance of noise
  posterior_sigma <- sqrt(posterior_sigma2)         # Posterior of sd of noise

  # Noise variance
  # -- Old code ---
  # posterior_noise <- matrix(rep(NA, n.samples*(pred_size-1)), ncol=(pred_size-1), nrow=n.samples)
  # for (i in 1:nrow(posterior_noise)){
  #   posterior_noise[i,] <- rnorm((pred_size-1), mean=0, sd=sqrt(posterior_sigma2[i]))
  # }
  # --- New code ---
  posterior_noise <- t(sapply(posterior_sigma, function(sd) {
    rnorm((pred_size-1), mean = 0, sd = sd)
  }))

  ess <- rfm_model$ess

  # Vector of x prime predictions
  x_prime_pred <- rep(NA, pred_size)
  x_prime_pred[1] <- if (pred_type=="mean"){
    mean(preds$full_samples)
  } else if (pred_type=="median"){
    median(preds$full_samples)
  }

  future_x_prime_preds <- matrix(NA, n.samples, pred_size) # matrix for x prime predictions
  future_x_prime_preds[,1] <- preds$full_samples

  y_pred <- rep(NA, pred_size) # vector for y values predictions

  future_y_preds <- matrix(NA, n.samples, pred_size)

  past <- rep(x = tail(train_dat_vec, 1), times = length(future_x_prime_preds[,1]))
  future_y_preds[,1] <- taken_theorem_predict(rate_of_change = future_x_prime_preds[,1] , past_values = past, time_step = pred_time_step[1])

  y_pred[1] <- if (pred_type=="mean"){
    mean(as.vector(future_y_preds[,1]))
  } else {
    median(as.vector(future_y_preds[,1]))
  }

  # Loop to generate multi-step ahead predictions
  # --- New code ---
  results <- Reduce(f = function(state, i) {
    # unpack previous state
    future_y_preds <- state$future_y_preds
    future_x_prime_preds <- state$future_x_prime_preds
    x_prime_pred <- state$x_prime_pred
    y_pred <- state$y_pred

    # new column i
    future_x_prime_preds[, i] <- vapply(1:n.samples, function(j) {
      fit_data <- tail(c(train_dat_vec, future_y_preds[j, 1:(i - 1)]), window_size)
      x_test <- matrix(as.numeric(fit_data), nrow = 1, ncol = window_size, byrow = TRUE)
      Z_test <- transformation_and_activation(x = x_test, W = W, b = b, act_func = act_func)
      pred <- cbind(1, Z_test) %*% t(matrix(posterior_samples[j, ], nrow=1, ncol=(ncol(Z_test)+1), byrow=TRUE)) + posterior_noise[j, (i - 1)]
      pred
    }, numeric(1)) # expect the function to return a single numeric value for each j

    # aggregate point forecast
    x_prime_pred[i] <- if (pred_type == "mean") {
      mean(future_x_prime_preds[, i])
    } else {
      median(as.vector(future_x_prime_preds[, i]))
    }

    future_y_preds[, i] <- taken_theorem_predict(
      rate_of_change = future_x_prime_preds[, i],
      past_values = future_y_preds[, (i - 1)],
      time_step = pred_time_step[i]
    )

    y_pred[i] <- if (pred_type == "mean") {
      mean(future_y_preds[, i])
    } else {
      median(as.vector(future_y_preds[, i]))
    }

    # return updated state
    list(
      future_y_preds = future_y_preds,
      future_x_prime_preds = future_x_prime_preds,
      x_prime_pred = x_prime_pred,
      y_pred = y_pred
    )
  }, x = 2:pred_size, init = list(
    future_y_preds = future_y_preds,
    future_x_prime_preds = future_x_prime_preds,
    x_prime_pred = x_prime_pred,
    y_pred = y_pred
  ))

  # --- Old code ---
  # for (i in 2:pred_size) {
  #   for (j in 1:n.samples){
  #     fit_data <- tail(c(train_dat_vec, as.vector(future_y_preds[j, 1:(i - 1)])), window_size)
  #     x_test <- matrix(as.numeric(as.vector(fit_data)), nrow = 1, ncol = window_size, byrow = TRUE)
  #
  #     # Transformation
  #     Z_test <- transformation_and_activation(x = x_test, W = W, b = b, act_func = act_func)
  #
  #     # Perform matrix multiplication for predictions
  #     future_x_prime_preds[j, i] <- cbind(rep(1, nrow(Z_test)), Z_test) %*% t(matrix(posterior_samples[j, ], nrow=1, ncol=(ncol(Z_test)+1), byrow=TRUE)) + posterior_noise[j, (i-1)]
  #   }
  #
  #   if (pred_type=="mean"){
  #     x_prime_pred[i] <- mean(as.vector(future_x_prime_preds[, i]))
  #   } else if (pred_type=="median"){
  #     x_prime_pred[i] <- median(as.vector(future_x_prime_preds[, i]))
  #   }
  #
  #   future_y_preds[,i] <- taken_theorem_predict(rate_of_change = future_x_prime_preds[, i], past_values = future_y_preds[, (i-1)], time_step = pred_time_step)
  #
  #   if (pred_type=="mean"){
  #     y_pred[i] <- mean(as.vector(future_y_preds[, i]))
  #   } else if (pred_type=="median"){
  #     y_pred[i] <- median(as.vector(future_y_preds[, i]))
  #   }
  # }

  alpha.CI = (100-CI)/100
  x_prime_pred_ci_matrix = credible_interval(x = results$future_x_prime_preds, alpha = alpha.CI)
  y_pred_ci_matrix = credible_interval(x = results$future_y_preds, alpha = alpha.CI)
  smooth_var = NULL
  smooth_diff = NULL
  raw_diff = rate_of_change

  return(list(
    fit_results = fit_model,
    fitted.values = fitted.values,
    W = W,
    b = b,
    # DIC = DIC,
    WAIC = WAIC,
    # modelWAIC = modelWAIC,

    # x_prime_pred = results$x_prime_pred,
    y_pred = results$y_pred,

    # future_x_prime_preds = results$future_x_prime_preds, # for constructing credible interval for x prime
    # xprime.pred.ci = x_prime_pred_ci_matrix,             # credible interval
    # xprime.pred.lower.ci = x_prime_pred_ci_matrix[,1],
    # xprime.pred.upper.ci = x_prime_pred_ci_matrix[,2],

    future_y_preds = results$future_y_preds,     # for constructing credible interval for predictions
    pred.ci = y_pred_ci_matrix,                  # credible interval
    # pred.lower.ci = y_pred_ci_matrix[,1],
    # pred.upper.ci = y_pred_ci_matrix[,2],

    # n_features = n_features,
    posterior_samples = posterior_samples,       # samples of coefficients including intercepts
    posterior_sigma2 = posterior_sigma2,         # samples of variance of errors
    ess = ess,

    raw_diff = raw_diff,
    smooth_diff = smooth_diff,
    var.smooth = smooth_var
  ))
}

# Bayesian Taken with Smoothed Rate of Change and Corrected Error Sampling from a Normal Distribution
ts_forecast_bayes_taken_reg_rfm_smooth <- function(ts_data,
                                                   delta_t,
                                                   sdiff = NULL,
                                                   raw_diff = NULL,
                                                   window_size = 9,
                                                   pred_size = 7,
                                                   pred_time_step = 1,
                                                   weight_dist = "normal",
                                                   weight_params = list(),
                                                   bias_dist = "uniform",
                                                   bias_params = list(min_val=0, max_val=2*pi),
                                                   act_func = "fourier",
                                                   reg_type = "lasso",
                                                   family = "gaussian",
                                                   burnin = 1000,
                                                   n.samples = 2000,
                                                   thin = 5,
                                                   n.cores = 1,
                                                   pred_type = "mean",
                                                   feature_selection = "sqrt",
                                                   feature_constant = NULL,
                                                   CI = 95) {

  if (length(sdiff) != length(raw_diff)){
    warning("`smooth_rate_of_change` length does not match with foward difference (rate of change) calculated by the package.")
  }

  train_dat_vec <- as.vector(ts_data)
  # Embed the training data
  embed_dat_train <- embed(train_dat_vec, dimension = window_size + 1)
  if (nrow(embed_dat_train) > length(sdiff[window_size:length(sdiff)])){
    stop("Please check the size of the input derivative.")
  }
  x_train <- embed_dat_train[, 2:(window_size + 1)]

  # Reverse the data in time order
  x_train <- x_train[,window_size:1]

  # Target variable is the smoothing derivative
  y_train <- sdiff[window_size:(nrow(x_train) + window_size - 1)] # consider appropriate index, if it is backward derivative then it is different
  delta_t_train <- delta_t[window_size:(nrow(x_train) + window_size - 1)] # appropriate index of training delta t
  raw_diff_train <- raw_diff[window_size:(nrow(x_train) + window_size - 1)]

  # Variance of error due to smoothing the rate of change
  smooth_var <- var(raw_diff_train-y_train)

  if (length(y_train) != length(raw_diff_train)){
    stop("Check length of smoothing function.")
  }
  if (nrow(x_train) != length(y_train)){
    stop("Please check dimensions of x_train and y_train.")
  }
  x_test <- matrix(rev(embed_dat_train[nrow(embed_dat_train), 1:window_size]), nrow = 1, byrow = TRUE) # reverse the data

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
  preds <- predict.BayesRandFeatRegModel(rfm_model, newx = x_test, CI = CI)

  fit_model <- rfm_model$fit_results                # Fitted GLM model
  W <- rfm_model$W                                  # Random feature weights
  b <- rfm_model$b                                  # Random feature biases
  # DIC <- rfm_model$DIC
  WAIC <- rfm_model$WAIC
  # modelWAIC <- rfm_model$modelWAIC
  x_derivative_train_preds <- rfm_model$train_preds # Predictions on training set, the derivative predictions
  fitted.values <- x_derivative_train_preds[,2] * delta_t_train + as.vector(x_train[, window_size])

  posterior_samples <- rfm_model$posterior_samples  # Posterior of coefficients
  posterior_sigma2 <- rfm_model$posterior_sigma2    # Posterior of variance of noise
  posterior_sigma <- sqrt(posterior_sigma2)

  # Noise variance
  # --- New code ---
  posterior_noise <- t(sapply(posterior_sigma, function(sd) {
    rnorm((pred_size-1), mean = 0, sd = sd)
  }))
  # --- Old code ---
  # posterior_noise <- matrix(rep(NA, n.samples*(pred_size-1)), ncol=(pred_size-1), nrow=n.samples)
  # for (i in 1:nrow(posterior_noise)){
  #   posterior_noise[i,] <- rnorm((pred_size-1), mean=0, sd=sqrt(posterior_sigma2[i]))
  # }

  ess <- rfm_model$ess

  smooth_x_prime_pred <- rep(NA, pred_size) # vector of x prime predictions
  x_prime_pred <- rep(NA, pred_size) # vector of x prime predictions

  future_smooth_x_prime_preds <- matrix(NA, n.samples, pred_size) # matrix for x prime predictions
  future_smooth_x_prime_preds[,1] <- preds$full_samples
  smooth_x_prime_pred[1] <- if (pred_type=="mean"){
    mean(preds$full_samples)
  } else {
    median(preds$full_samples)
  }

  # Use error from normal distribution
  smooth_error_samples <- matrix(rnorm(n.samples*pred_size, mean=0, sd=sqrt(smooth_var)), nrow=n.samples, ncol=pred_size)

  future_x_prime_preds <- matrix(NA, n.samples, pred_size) # matrix for x prime predictions
  future_x_prime_preds[,1] <- future_smooth_x_prime_preds[,1]+smooth_error_samples[,1]
  x_prime_pred[1] <- if (pred_type=="mean"){
    mean(as.vector(future_x_prime_preds[,1]))
  } else {
    median(as.vector(future_x_prime_preds[,1]))
  }

  y_pred <- rep(NA, pred_size) # vector for y values predictions

  future_y_preds <- matrix(NA, n.samples, pred_size)

  past <- rep(x=tail(train_dat_vec, 1), times=length(future_x_prime_preds[,1]))
  future_y_preds[,1] <- taken_theorem_predict(rate_of_change=future_x_prime_preds[,1], past_values=past, time_step=pred_time_step[1])

  y_pred[1] <- if (pred_type=="mean"){
    mean(as.vector(future_y_preds[,1]))
  } else if (pred_type=="median"){
    median(as.vector(future_y_preds[,1]))
  }

  # Loop to generate multi-step ahead predictions
  # --- New code ---
  results <- Reduce(f = function(state, i) {
    # unpack previous state
    future_y_preds <- state$future_y_preds
    future_x_prime_preds <- state$future_x_prime_preds
    future_smooth_x_prime_preds <- state$future_smooth_x_prime_preds

    y_pred <- state$y_pred
    x_prime_pred <- state$x_prime_pred
    smooth_x_prime_pred <- state$smooth_x_prime_pred

    # can be optimized
    future_smooth_x_prime_preds[, i] <- vapply(1:n.samples, function(j) {
      fit_data <- tail(c(train_dat_vec, future_y_preds[j, 1:(i - 1)]), window_size)
      x_test <- matrix(as.numeric(fit_data), nrow = 1, ncol = window_size, byrow = TRUE)
      Z_test <- transformation_and_activation(x = x_test, W = W, b = b, act_func = act_func)
      pred <- cbind(1, Z_test) %*% t(matrix(posterior_samples[j, ], nrow=1, ncol=(ncol(Z_test)+1), byrow=TRUE)) + posterior_noise[j, (i - 1)]
      pred
    }, numeric(1)) # expect the function to return a single numeric value for each j

    smooth_x_prime_pred[i] <- if (pred_type=="mean"){
      mean(as.vector(future_smooth_x_prime_preds[, i]))
    } else {
      median(as.vector(future_smooth_x_prime_preds[, i]))
    }

    future_x_prime_preds[,i] <- future_smooth_x_prime_preds[,i]+smooth_error_samples[,i]
    x_prime_pred[i] <- if (pred_type=="mean"){
      mean(as.vector(future_x_prime_preds[, i]))
    } else {
      median(as.vector(future_x_prime_preds[, i]))
    }

    future_y_preds[, i] <- taken_theorem_predict(
      rate_of_change = future_x_prime_preds[, i],
      past_values = future_y_preds[, (i - 1)],
      time_step = pred_time_step[i]
    )

    y_pred[i] <- if (pred_type == "mean") {
      mean(future_y_preds[, i])
    } else {
      median(as.vector(future_y_preds[, i]))
    }

    # return updated state
    list(
      future_y_preds = future_y_preds,
      future_x_prime_preds = future_x_prime_preds,
      future_smooth_x_prime_preds = future_smooth_x_prime_preds,

      y_pred = y_pred,
      x_prime_pred = x_prime_pred,
      smooth_x_prime_pred = smooth_x_prime_pred
    )
  }, x = 2:pred_size, init = list(
    future_y_preds = future_y_preds,
    future_x_prime_preds = future_x_prime_preds,
    future_smooth_x_prime_preds = future_smooth_x_prime_preds,

    y_pred = y_pred,
    x_prime_pred = x_prime_pred,
    smooth_x_prime_pred = smooth_x_prime_pred
  ))
  # --- Old code ---
  # Loop to generate multi-step ahead predictions
  # for (i in 2:pred_size) {
  #   for (j in 1:n.samples){
  #     fit_data <- tail(c(train_dat_vec, as.vector(future_y_preds[j, 1:(i - 1)])), window_size)
  #     x_test <- matrix(as.numeric(as.vector(fit_data)), nrow = 1, ncol = window_size, byrow = TRUE)
  #
  #     # Transformation
  #     Z_test <- transformation_and_activation(x = x_test, W = W, b = b, act_func = act_func)
  #
  #     # Perform matrix multiplication for predictions
  #     future_smooth_x_prime_preds[j, i] <- cbind(rep(1, nrow(Z_test)), Z_test)%*%t(matrix(posterior_samples[j, ], nrow=1, ncol=(ncol(Z_test)+1), byrow=TRUE))+posterior_noise[j, (i-1)]
  #   }
  #
  #   if (pred_type=="mean"){
  #     smooth_x_prime_pred[i] <- mean(as.vector(future_smooth_x_prime_preds[, i]))
  #   } else if (pred_type=="median"){
  #     smooth_x_prime_pred[i] <- median(as.vector(future_smooth_x_prime_preds[, i]))
  #   }
  #   smooth_error_samples[,i] <- rnorm(n.samples, mean=0, sd=sqrt(smooth_var))
  #   future_x_prime_preds[,i] <- future_smooth_x_prime_preds[,i]-smooth_error_samples[,i]
  #   if (pred_type=="mean"){
  #     x_prime_pred[i] <- mean(as.vector(future_x_prime_preds[, i]))
  #   } else if (pred_type=="median"){
  #     x_prime_pred[i] <- median(as.vector(future_x_prime_preds[, i]))
  #   }
  #
  #   future_y_preds[,i] <- taken_thm_fwd_pred(derivative=future_x_prime_preds[, i], past_value=future_y_preds[, (i-1)], delta_t=pred_time_step)
  #
  #   if (pred_type=="mean"){
  #     y_pred[i] <- mean(as.vector(future_y_preds[, i]))
  #   } else if (pred_type=="median"){
  #     y_pred[i] <- median(as.vector(future_y_preds[, i]))
  #   }
  # }

  alpha.CI = (100-CI)/100
  smooth_x_prime_pred_ci_matrix = credible_interval(x=results$future_smooth_x_prime_preds, alpha=alpha.CI)
  x_prime_pred_ci_matrix = credible_interval(x=results$future_x_prime_preds, alpha=alpha.CI)
  y_pred_ci_matrix = credible_interval(x=results$future_y_preds, alpha=alpha.CI)

  return(list(
    fit_results = fit_model,
    fitted.values = fitted.values,
    W = W,
    b = b,
    # DIC = DIC,
    WAIC = WAIC,
    # modelWAIC = modelWAIC,

    # smooth_x_prime_pred = smooth_x_prime_pred,
    # x_prime_pred = x_prime_pred,
    y_pred = results$y_pred,

    # Derivative return
    # error.smooth = smooth_error,
    # error_samples.smooth = smooth_error_samples,
    # true_smooth_xprime_test = smooth_derivative_test,
    # true_xprime_test = raw_derivative_test,

    # future_smooth_x_prime_preds = future_smooth_x_prime_preds, # for constructing credible interval for x prime
    # smooth_xprime.pred.ci = smooth_x_prime_pred_ci_matrix,     # credible interval
    # smooth_xprime.pred.lower.ci = smooth_x_prime_pred_ci_matrix[,1],
    # smooth_xprime.pred.upper.ci = smooth_x_prime_pred_ci_matrix[,2],
    #
    # future_x_prime_preds = future_x_prime_preds, # for constructing credible interval for x prime
    # xprime.pred.ci = x_prime_pred_ci_matrix,     # credible interval
    # xprime.pred.lower.ci = x_prime_pred_ci_matrix[,1],
    # xprime.pred.upper.ci = x_prime_pred_ci_matrix[,2],

    future_y_preds = results$future_y_preds,     # for constructing credible interval for predictions
    pred.ci = y_pred_ci_matrix,                  # credible interval
    # pred.lower.ci = y_pred_ci_matrix[,1],
    # pred.upper.ci = y_pred_ci_matrix[,2],

    # n_features = n_features,
    posterior_samples = posterior_samples,       # samples of coefficients including intercepts
    posterior_sigma2 = posterior_sigma2,         # samples of variance of errors
    ess = ess,

    raw_diff = raw_diff,
    smooth_diff = sdiff,
    var.smooth = smooth_var
  ))
}

#' Bayesian Regularization Regression with Random Feature Models for Delay Embedding
#'
#' This function aims to forecast time series data for a specified number of time steps (`pred_size`) using Bayesian regularization regression with random features model for Delay Embedding.
#' Currently, only Gaussian distribution is supported.
#' The `bayesreg` package is used to estimate the model coefficients via penalized regression.
#'
#' @param ts_data A numeric vector representing the time series data.
#' @param time A numeric vector representing the time gap between two consecutive observations in time series. Defaults to `NULL`.
#' @param smooth_diff A boolean value for smoothing the time derivatives.
#'  Possible values: `TRUE` (default), `T`, `FALSE`, `F`.
#' @param method A character string of the smooth method for the time derivatives.
#'  Defaults to `ma` for the right-aligned moving average lags. Other supported options include `"polynomial"`, `"localized_polynomial"`, `"spline"`.
#' @param smooth_params A list of the parameters for the smooth function.
#'   - If `method = "ma"`, the list contains:
#'     - `window`: Window size to apply moving average. Defaults to `5`.
#'   - If `method = "polynomial"`, the list contains:
#'     - `polyorder`: The order of the polynomial. Defaults to `5`.
#'   - If `method = "localized_polynomial"`, the list contains:
#'     - `span`: The size how localized is the fit, which ranges between 0 to 1. Defaults to `0.5`.
#'   - If `method = "spline"`, the list contains:
#'     - `knotnum`: The number of knots. Defaults to `5`.
#'
#'   Defaults to an empty list `list()` (which is interpreted differently based on `method`).
#' @param window_size An integer specifying the embedding window size (i.e., the number of past observations used as features). Defaults to `9`.
#' @param pred_size An integer specifying the number of future time steps to predict. Defaults to `7`.
#' @param pred_time_step An numeric value or a vector of time step for forecasting. Defaults to `1`.
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
#' \bar{\mathbf{y}}'\gets
#' \begin{bmatrix}
#'    \bar{y}'_{t_m}\\
#'    \bar{y}'_{t_{m+1}}\\
#'    \vdots\\
#'    \bar{y}'_{t_{n-1}}
#' \end{bmatrix}
#' }
#' where \eqn{\bar{y}'_{t_k}} is calculated using smooth function defined by `smooth_params` when `smooth_diff = c(TRUE, T)`
#' with the input of \deqn{y'_{t_k} \approx \frac{y_{t_{k+1}}-y_{t_k}}{t_{k+1}-t_k}} for \eqn{k=1, \dots, n-1}.
#'
#' The model is defined as:
#' \deqn{\bar{\mathbf{y}}' = \beta_0+\sigma(\mathbf{XW} + \mathbf{b}) \boldsymbol{\beta}+\epsilon}
#' where \eqn{\sigma} is the activation function, \eqn{\mathbf{X}} is the input matrix as defined above, \eqn{\mathbf{W}} is the random weighting matrix, \eqn{\mathbf{b}} is the random bias vector,
#' \eqn{\beta_0} is the estimated intercept, \eqn{\boldsymbol{\beta}} is the learnt weights matrix using regularization techniques, and \eqn{\epsilon\sim \mathcal{N}(0, \sigma_{\epsilon}^2)} represents errors.
#'
#' Note that \eqn{\beta_0}, \eqn{\boldsymbol{\beta}}, and \eqn{\sigma_{\epsilon}^2} are estimated using \code{bayesreg::bayesreg}.
#'
#' @return A list including predicted values and model output generated using Takens' theorem, based on the Bayesian regularization regression with random features model,
#' specifying for time series data with the smoothness rate of change:
#'
#' \code{fit_results}: The result of fitted model from \code{bayesreg::bayesreg}.
#'
#' \code{fitted.values}: The fitted mean or median values defined by `pred_type`.
#'
#' \code{W}: The matrix of random weights used for random features model.
#'
#' \code{b}: The vector of random biases used for random features model.
#'
#' \code{WAIC}: The Watanabe–Akaike information criterion of the fitted model.
#'
#' \code{y_pred}: A numeric vector of forecast values for future time steps, with length equal to \code{pred_size}.
#'
#' \code{future_y_preds}: A matrix of all posterior predictive samples for future time steps.
#'
#' \code{pred.ci}: A matrix including the credible interval based on the input of \code{CI}.
#'
#' \code{posterior_samples}: A matrix of posterior samples of model coefficients.
#'
#' \code{posterior_sigma2}: A numeric vector of posterior samples of the error variance.
#'
#' \code{ess}: An \code{"ess_plot"} object including effective sample sizes (ESS) for MCMC diagnostics.
#'
#' \code{raw_diff}: A vector of the time derivatives.
#'
#' \code{smooth_diff}: A vector of the smooth time derivatives.
#'
#' \code{var.smooth}: The variance of error due to smoothing effect.
#'
#' @examples
#' \dontrun{
#' set.seed(456)
#' y = cumsum(rnorm(1000))
#' fitted_bayes_reg_rfm_taken <- ts_forecast_bayes_reg_rfm_taken(ts_data=y)
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
ts_forecast_bayes_reg_rfm_taken <- function(ts_data,
                                            time = NULL, # in numeric format
                                            smooth_diff = TRUE,
                                            method = "ma",
                                            smooth_params = list(),
                                            window_size = 9,
                                            pred_size = 7,
                                            pred_time_step = 1,
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
                                            CI = 95){
  # Check input
  if (is.null(ts_data)) stop("Please check your input time series.")
  if (is.null(time)) time <- 1:length(ts_data)
  if (length(ts_data) != length(time)) stop("Time series must have the same length: ", length(ts_data), " vs ", length(time))

  if (!(smooth_diff %in% c(TRUE, FALSE, T, F))) stop("`smooth_diff` must be a boolean value.")
  if (!(positive_int_check(window_size))) stop("`window_size` must be a positive integer.")
  if (!(positive_int_check(pred_size))) stop("`pred_size` must be a positive integer.")
  if (!(positive_int_check(pred_time_step))) stop("`pred_time_step` must be a positive integer.")
  if (!(reg_type %in% c("lasso", "ridge"))) stop("`reg_type` is invalid. Please choose `lasso` or `ridge`.")

  if (window_size > length(ts_data)) stop("`window_size` exceeds `length(ts_data)`.")

  if (length(pred_time_step) == 1) pred_time_step <- rep(pred_time_step, pred_size)
  if (length(pred_time_step) != pred_size) stop("Please check your input of `pred_time_step`.")

  if (!(pred_type %in% c("mean", "median"))) stop("`pred_type` is invalid. Please choose 'mean' or 'median'.")

  # Currently support
  # *Error distribution: Gaussian distribution
  family = "gaussian"
  # *Finite difference: forward difference
  fdm = "forward"
  delta_t = diff(time, 1)

  # Compute rate of change
  if (fdm == "forward") raw_diff = na.omit(forward_difference(x=as.vector(ts_data), step=delta_t))

  if (smooth_diff %in% c(TRUE, T)){
    sdiff <- na.omit(smoothness_ts(ts_data=as.vector(raw_diff), fn_name=method, params=smooth_params))

    return(
      ts_forecast_bayes_taken_reg_rfm_smooth(
        ts_data = ts_data,
        delta_t = delta_t,
        sdiff = sdiff,
        raw_diff = raw_diff,
        window_size = window_size,
        pred_size = pred_size,
        pred_time_step = pred_time_step,
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
        feature_selection = feature_selection,
        feature_constant = feature_constant,
        CI = CI
      )
    )

  } else {
    return(
      ts_forecast_bayes_taken_reg_rfm_no_smooth(
        ts_data = ts_data,
        delta_t = delta_t,
        rate_of_change = raw_diff,
        window_size = window_size,
        pred_size = pred_size,
        pred_time_step = pred_time_step,
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
        feature_selection = feature_selection,
        feature_constant = feature_constant,
        CI = CI
      )
    )
  }
}
