#' Fit frequentist regularization regression with random features
#'
#' The function aims to fit the regularization random features model using the frequentist approach.
#' Currently, only Gaussian distribution is supported.
#' The `glmnet` package is used to estimate the model coefficients via penalized regression.
#'
#' @param x A matrix of features.
#' @param y A response vector.
#' @param n_features An integer of the number of random features. Defaults to `100`.
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
#' @param standardize A logical flag to standardize the transformation of the random features model for `glmnet::glmnet()`. Defaults to `TRUE`.
#' @param nfolds An integer of the number of folds. Defaults to `10`.
#' @param reg_type A character string of the regularization method.
#'  Possible values: `"lasso"` (default), `"ridge"`, `"elastic net"`.
#' @param alpha A numeric value for the elastic net mixing parameter.
#'   Use `alpha = 1` for LASSO, `alpha = 0` for Ridge, `alpha = 0.5` (default) for Elastic net.
#'   If `reg_type` is `"lasso"` or `"ridge"`, `alpha` can be set to `NULL`.
#'   Defaults to `NULL`.
#' @param lambda_type A character string indicating the type of lambda to use after cross-validation.
#'   Possible values are `"min"` (default), and `"1se"`.
#' @param lambda A numeric vector of possible lambda values to use in cross-validation. Defaults to `c(10^-3, 10^-4, 10^-5, 10^-6, 10^-7)`.
#'
#' @details
#' The model is defined as:
#' \deqn{\mathbf{y} = \beta_0+\sigma(\mathbf{XW} + \mathbf{b}) \boldsymbol{\beta}+\epsilon}
#' where \eqn{\sigma} is the activation function, \eqn{\mathbf{X}} is the input matrix, \eqn{\mathbf{W}} is the random weighting matrix, \eqn{\mathbf{b}} is the random bias vector,
#' \eqn{\beta_0} is the estimated intercept, \eqn{\boldsymbol{\beta}} is the learnt weights matrix using regularization techniques, and \eqn{\epsilon\sim \mathcal{N}(0, \sigma_{\epsilon}^2)} represents errors.
#'
#' Note that \eqn{\beta_0} and \eqn{\boldsymbol{\beta}} are estimated using \code{glmnet::glmnet}.
#'
#' @return A list of class `"FreqRandFeatRegModel"` containing the fitted model with components::
#'
#' \code{fit_results}: The result of fitted model from \code{glmnet::glmnet}.
#'
#' \code{lambda}: The optimal lambda value obtained from \code{glmnet::cv.glmnet}.
#'
#' \code{W}: The matrix of random weights used of random features model.
#'
#' \code{b}: The vector of random biases used of random features model.
#'
#' \code{AIC}: The Akaike Information Criterion of the fitted model.
#'
#' \code{cv_model}: The result of cross-validation model from \code{glmnet::cv.glmnet}.
#'
#' \code{train_preds}: A numeric vector of predicted values for the training data.
#'
#' @examples
#' \dontrun{
#' set.seed(456)
#' x = cbind(rnorm(1000), runif(1000, min = -1, max = 1))
#' y = x[,1] + 2*x[,2] + x[,1]^2 + x[,1]^3
#'
#' fitted <- fit_freq_reg_rfm(x, y, n_features = 50)
#' }
#'
#' @import glmnet
#'
#' @export
#'
#' @references
#' \code{glmnet}: Lasso and Elastic-Net Regularized Generalized Linear Models
#'
#' CRAN package manual: \url{https://cran.r-project.org/web/packages/glmnet/index.html}
#'
fit_freq_reg_rfm <- function(x,
                             y,
                             n_features = 100,
                             weight_dist = "normal",
                             weight_params = list(),
                             bias_dist = "uniform",
                             bias_params = list(min_val = 0, max_val = 2*pi),
                             act_func = "fourier",
                             standardize = TRUE,
                             nfolds = 10,
                             reg_type = "lasso",
                             alpha = NULL,
                             lambda_type = "min",
                             lambda = c(10^-3, 10^-4, 10^-5, 10^-6, 10^-7))
{
  # Check for input
  if (is.null(x) | is.null(y)) stop("Please check input data.")
  # Ensure that alpha is properly set for `elastic net`, `lasso`, and `ridge`
  if (!is.null(alpha) && (alpha < 0 || alpha > 1)) stop("`alpha` must be between 0 and 1.")
  if (reg_type == "lasso" && !is.null(alpha) && alpha != 1) stop("'lasso' requires alpha equals 1.")
  if (reg_type == "ridge" && !is.null(alpha) && alpha != 0)  stop("'ridge' requires alpha equals 0.")

  if (!(positive_int_check(n_features))) stop("`n_features` must be a positive integer.")
  if (!(positive_int_check(nfolds))) stop("`nfolds` must be a positive integer.")

  # Current support
  family = "gaussian"

  # Transform training data using random features model (RFM)
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

  rf_generator <- function(newx) transformation_and_activation(newx, W_train, b_train, act_func)

  # Determine alpha based on regularization type
  reg_alpha <- switch(
    reg_type,
    "ridge" = 0,
    "lasso" = 1,
    "elastic net" = ifelse(!is.null(alpha), alpha, 0.5),
    stop(
      "Invalid regularization type specified. Please choose 'ridge', 'lasso', or 'elastic net'."
    )
  )

  # Perform cross-validation to select optimal lambda
  cv_model <- glmnet::cv.glmnet(
    Z_train,
    y,
    alpha = reg_alpha,
    lambda = lambda,
    standardize = standardize,
    nfolds = nfolds,
    family = family
  )

  chosen_lambda <- switch(lambda_type,
                          "min" = cv_model$lambda.min,
                          "1se" = cv_model$lambda.1se,
                          stop("Invalid lambda_type specified. Please choose lambda 'min' or '1se'."))

  # Fit the best model with the chosen lambda
  best_model <- glmnet::glmnet(
    Z_train,
    y,
    alpha = reg_alpha,
    lambda = chosen_lambda,
    standardize = standardize,
    family = family
  )

  # Generate predictions on training set
  train_preds <- predict(best_model, newx = Z_train)

  # AIC
  if (family == "gaussian") {
    log_lik <- log_likelihood_norm_vec(y = y, y_pred = train_preds)

    # Adjust df to include intercept
    k <- best_model$df + 1
  }

  # Compute AIC
  aic_value <- aic_function(log_lik = log_lik, k = k)

  model <- list(
    fit_results = best_model,
    lambda = chosen_lambda,
    W = W_train,
    b = b_train,
    AIC = aic_value,
    cv_model = cv_model,
    train_preds = train_preds,
    rf_generator = rf_generator
  )
  class(model) <- "FreqRandFeatRegModel"
  return(model)
}

#' Predict function for FreqRandFeatRegModel objects
#'
#' The function aims to predict the response value given a new input newx using Random Features Model under the Frequentist Regularization approach.
#'
#' @param object A FreqRandFeatRegModel object.
#' @param newx New input data (a matrix) for prediction.
#' @param ... Additional arguments (ignored).
#'
#' @return A vector of predicted values.
#'
#' @export
#'
# Fit regularized random features model on test set.
# Apply for independent observations data model.
predict.FreqRandFeatRegModel <- function(object, newx, ...) {
  if (is.null(object$rf_generator)) stop("Missing transformation function.")
  Z_new <- object$rf_generator(newx)
  preds <- predict(object$fit_results, newx = Z_new, s = object$lambda)
  return(as.vector(preds))
}


#' Forecasting for time series using regularization regression with random features under frequentist framework
#'
#' The frequentist regularization regression with random feature has been applied to predict the values of the time series with the specifying value of time steps.
#' Currently, only Gaussian distribution is supported.
#' The `glmnet` package is used to estimate the model coefficients via penalized regression.
#'
#' @param ts_data A numeric vector representing the time series data.
#' @param window_size A positive integer specifying the embedding window size (i.e., the number of past observations used as features). Defaults to `9`.
#' @param pred_size A positive integer specifying the number of future time steps to predict. Defaults to `7`.
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
#' @param standardize A logical flag to standardize the transformation of the random features model for `glmnet::glmnet()`. Defaults to `TRUE`.
#' @param nfolds An integer of the number of folds. Defaults to `10`.
#' @param alpha A numeric value for the elastic net mixing parameter.
#'   Use `alpha = 1` for LASSO, `alpha = 0` for Ridge, `alpha = 0.5` (default) for Elastic net.
#'   If `reg_type` is `"lasso"` or `"ridge"`, `alpha` can be set to `NULL`.
#'   Defaults to `NULL`.
#' @param lambda A numeric vector of possible lambda values to use in cross-validation. Defaults to `c(10^-3, 10^-4, 10^-5, 10^-6, 10^-7)`.
#' @param lambda_type A character string indicating the type of lambda to use after cross-validation.
#'   Possible values are `"min"` (default), and `"1se"`.
#' @param reg_type A character string of the regularization method.
#'   Possible values: `"lasso"` (default), `"ridge"`, `"elastic net"`.
#' @param feature_selection A character string specifying the method for selecting the number of random features.
#'   Possible values are:
#'   - `"sqrt"` (default): The number of features is set to the square root of the number of rows of training matrix `nrow`, i.e., `sqrt(nrow)`.
#'   - `"factor"`: The number of features is `feature_constant*nrow`.
#'   - `"constant"`: The number of features is fixed at `feature_constant`.
#'   Note that the number of features will be always rounded down.
#' @param feature_constant A numeric value used when `feature_selection` is `"factor"` or `"constant"`.
#'   Ignored when `feature_selection = "sqrt"`. Defaults to `NULL`.
#' @param bootstrap_CI A boolean value for constructing confidence intervals using bootstrapping Defaults to `FALSE`.
#' @param bootstrap_ref_point A reference point in the training time series for bootstrapping Defaults to `NULL`.
#'   If `bootstrap_CI = TRUE`. `bootstrap_ref_point = length(ts_data)/3` (default) if it is greater than `pred_size`, otherwise it is `length(ts_data)-pred_size`.
#' @param CI A numeric value indicating the level of the confidence interval to report (in percentage). Defaults to `95`.
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
#' where \eqn{\sigma} is the activation function, \eqn{\mathbf{X}} is the input matrix as defined above, \eqn{\mathbf{W}} is the random weighting matrix, \eqn{\mathbf{b}} is the random bias vector,
#' \eqn{\beta_0} is the estimated intercept, \eqn{\boldsymbol{\beta}} is the learnt weights matrix using regularization techniques, and \eqn{\epsilon\sim \mathcal{N}(0, \sigma_{\epsilon}^2)} represents errors.
#'
#' Note that \eqn{\beta_0} and \eqn{\boldsymbol{\beta}} are estimated using \code{glmnet::glmnet}.
#'
#' @return A list containing the fitted model and relevant components:
#'
#' \code{fit_results}: The result of fitted model from \code{glmnet::glmnet}.
#'
#' \code{cv_model}: The result of cross-validation model from \code{glmnet::cv.glmnet}.
#'
#' \code{W}: The matrix of random weights used of random features model.
#'
#' \code{b}: The vector of random biases used of random features model.
#'
#' \code{y_pred}: A numeric vector of forecast values for future time steps, with length equal to \code{pred_size}.
#'
#' \code{pred.ci}: A matrix of the confidence interval using bootstrapping.
#'
#' \code{AIC}: The Akaike Information Criterion of the fitted model.
#'
#' \code{train_preds}: A numeric vector of predicted values for the training data, generated using the embedding window (\code{window_size}).
#'
#' \code{n_features}: A positive integer value of the number of random features.
#'
#' @examples
#' \dontrun{
#' set.seed(456)
#' y = cumsum(rnorm(1000))
#' fitted_freq_reg_rfm <- ts_forecast_freq_reg_rfm(
#'       ts_data = y, bootstrap_CI=TRUE,
#'       bootstrap_ref_point = 900)
#' }
#'
#' @import glmnet
#'
#' @export
#'
#' @references
#' \code{glmnet}: Lasso and Elastic-Net Regularized Generalized Linear Models
#'
#' CRAN package manual: \url{https://cran.r-project.org/web/packages/glmnet/index.html}
#'
ts_forecast_freq_reg_rfm <- function(ts_data,
                                     window_size = 9,
                                     pred_size = 7,
                                     weight_dist = "normal",
                                     weight_params = list(),
                                     bias_dist = "uniform",
                                     bias_params = list(min_val = 0, max_val = 2*pi),
                                     act_func = "fourier",
                                     standardize = TRUE,
                                     nfolds = 10,
                                     alpha = NULL,
                                     lambda = c(10^-3, 10^-4, 10^-5, 10^-6, 10^-7),
                                     lambda_type = "min",
                                     reg_type = "lasso",
                                     feature_selection = "sqrt",
                                     feature_constant = NULL,
                                     bootstrap_CI = FALSE,
                                     bootstrap_ref_point = NULL,
                                     CI = 95) {
  # Check input data
  if (is.null(ts_data) || !is.vector(ts_data)) stop("Please input a data vector.")

  if (!(positive_int_check(window_size))) stop("`window_size` must be a positive integer.")
  if (window_size > length(ts_data)) stop("`window_size` exceeds the length of `ts_data`.")

  if (!(positive_int_check(pred_size))) stop("`pred_size` must be a positive integer.")
  if (!(positive_int_check(nfolds))) stop("`nfolds` must be a positive integer.")

  if (!(bootstrap_CI %in% c(TRUE, FALSE, T, F))) stop("`bootstrap_CI` must be a boolean value.")
  if (!(is.null(bootstrap_ref_point)) && !(positive_int_check(bootstrap_ref_point))){
    stop("`bootstrap_ref_point` must be a positive integer.")
  }
  if (CI < 0 && CI > 100) stop("`CI` must be between 0 and 100.")

  # Current support
  family = "gaussian"

  # Compute SE for CI
  if (bootstrap_CI %in% c(TRUE, T)){
    res_se <- se_ts_freq_reg_rfm(
      ts_data = ts_data,
      ref_point = bootstrap_ref_point,
      pred_size = pred_size,
      window_size = window_size,
      weight_dist = weight_dist,
      weight_params = weight_params,
      bias_dist = bias_dist,
      bias_params = bias_params,
      act_func = act_func,
      standardize = standardize,
      nfolds = nfolds,
      reg_type = reg_type,
      alpha = alpha,
      lambda = lambda,
      lambda_type = lambda_type,
      feature_selection = feature_selection,
      feature_constant = feature_constant
    )
    se <- res_se$se # return of standard error using bootstrapping
  } else {
    se <- NULL
  }

  train_dat_vec <- ts_data # vector
  embed_dat_train <- embed(train_dat_vec, dimension = window_size + 1) # matrix
  # Extract columns for x_train
  x_train <- embed_dat_train[, 2:(window_size + 1)]
  # Reverse the data in time order
  x_train <- x_train[, window_size:1]

  # Target is the first column of the embed data
  y_train <- embed_dat_train[, 1]

  # Make sure that x_test is in matrix form
  x_test <- matrix(rev(embed_dat_train[nrow(embed_dat_train), 1:window_size]), nrow = 1, byrow = TRUE)

  # Determine number of features based on the selected method
  n_features <- check_num_feature(
    feature_selection = feature_selection,
    feature_constant = feature_constant,
    n = nrow(x_train)
  )

  rfm_model <- fit_freq_reg_rfm(
    x = x_train,
    y = y_train,
    n_features = n_features,
    weight_dist = weight_dist,
    weight_params = weight_params,
    bias_dist = bias_dist,
    bias_params = bias_params,
    act_func = act_func,
    standardize = standardize,
    nfolds = nfolds,
    reg_type = reg_type,
    alpha = alpha,
    lambda_type = lambda_type,
    lambda = lambda
  )

  W <- rfm_model$W
  b <- rfm_model$b
  fit_model <- rfm_model$fit_results
  cv_model <- rfm_model$cv_model
  train_preds <- rfm_model$train_preds
  AIC <- rfm_model$AIC

  y_pred <- rep(NA, pred_size)
  y_pred[1] <- predict.FreqRandFeatRegModel(rfm_model, x_test)

  # Use Reduce to update y_pred for the remaining steps
  y_pred <- Reduce(function(y_pred, i) {
    fit_data <- c(train_dat_vec, y_pred[1:(i - 1)])
    x_test <- matrix(
      tail(fit_data, window_size),
      nrow = 1,
      ncol = window_size,
      byrow = TRUE
    )

    # Predict using the fitted model
    y_pred[i] <- predict.FreqRandFeatRegModel(rfm_model, x_test)  # Update the i-th position in y_pred

    return(y_pred)  # Pass the updated y_pred to the next iteration
  }, 2:pred_size, init = y_pred)

  alpha = (100-CI)/100
  critical_value <- qnorm((alpha/2)+(CI/100))
  if (bootstrap_CI %in% c(TRUE, T)){
    pred.ci <- upper_lower_ci(pred=y_pred, se=se, pred_size=pred_size, critical_value=critical_value)
  } else {
    pred.ci <- NULL
  }


  return(
    list(
      fit_results = fit_model,
      cv_model = cv_model,
      W = W,
      b = b,
      y_pred = y_pred,
      pred.ci = pred.ci,
      AIC = AIC,
      train_preds = train_preds,
      n_features = n_features
    )
  )
}
