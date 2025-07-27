#' Forecast time series data using Takens' theorem under the frequentist regularization regression with random features model
#'
#' This function aims to forecast time series data for a specified number of time steps (`pred_size`) using frequentist regularization regression with random features model.
#' Currently, only Gaussian distribution is supported.
#' The `glmnet` package is used to estimate the model coefficients via penalized regression.
#'
#' @param ts_data A numeric vector representing the time series data.
#' @param time A numeric vector representing the time gap between two consecutive observations in time series. Defaults to `NULL`.
#' @param smooth_diff A boolean value for smoothing the time derivatives.
#'  Possible values: `TRUE` (default), `T`, `FALSE`, `F`.
#' @param method A character string of the smooth method for the time derivatives.
#'  Defaults to `ma` for the right-aligned moving average lags. Other supported options include `"polynomial"`, `"localized_polynomial"`, `"spline"`.
#' @param smooth_params A list of the parameters for the smooth function.
#'   - If `method = "ma"`, the list contains:
#'     - `window`: Window size to apply moving average. Defaults to `5`
#'   - If `method = "polynomial"`, the list contains:
#'     - `polyorder`: The order of the polynomial. Defaults to `5`
#'   - If `method = "localized_polynomial"`, the list contains:
#'     - `span`: The size how localized is the fit, which ranges between 0 to 1. Defaults to `0.5`.
#'   - If `method = "spline"`, the list contains:
#'     - `knotnum`: The number of knots. Defaults to `5`.
#'
#'   Defaults to an empty list `list()` (which is interpreted differently based on `method`).
#' @param window_size An integer specifying the embedding window size (i.e., the number of past observations used as features). Defaults to `9`.
#' @param pred_size An integer specifying the number of future time steps to predict. Defaults to `7`.
#' @param pred_time_step An integer or a vector of time gap between two consecutive observations.
#'   The vector must have length the same as `length(rate_of_change)`. Defaults to `1`.
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
#' @param alpha A numeric value for the elastic net mixing parameter.
#'   Use `alpha = 1` for LASSO, `alpha = 0` for Ridge, `alpha = 0.5` (default) for Elastic net.
#'   If `reg_type` is `"lasso"` or `"ridge"`, `alpha` can be set to `NULL`.
#'   Defaults to `NULL`.
#' @param lambda A numeric vector of possible lambda values to use in cross-validation. Defaults to `c(10^-3, 10^-4, 10^-5, 10^-6, 10^-7)`.
#' @param lambda_type A character string indicating the type of lambda to use after cross-validation.
#'   Possible values are `"min"` (default), and `"1se"`.
#' @param standardize A logical flag to standardize the transformation of the random feature model. Defaults to `TRUE`.
#' @param nfolds An integer of the number folds. Defaults to `10`.
#' @param reg_type A character string of the regularization method.
#'   Possible values: `"lasso"` (default), `"ridge"`, `"elastic net"`.
#' @param feature_selection A character string specifying the method for selecting the number of random features.
#'   Possible values are:
#'   - `"sqrt"` (default): The number of features is set to the square root of the number of rows of the training matrix `nrow`, i.e., `sqrt(nrow)`.
#'   - `"factor"`: The number of features is `feature_constant*nrow`.
#'   - `"constant"`: The number of features is fixed at `feature_constant`.
#'   Note that the number of features will be always rounded down.
#' @param feature_constant A numeric value used when `feature_selection` is `"factor"` or `"constant"`.
#'   Ignored when `feature_selection = "sqrt"`. Defaults to `NULL`.
#' @param bootstrap_CI A boolean value for constructing confidence intervals using bootstrapping. Defaults to `FALSE`.
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
#' Note that \eqn{\beta_0} and \eqn{\boldsymbol{\beta}} are estimated using \code{glmnet::glmnet}.
#'
#' The forecast value of the time series is computed as:
#' \deqn{y_{t_{k+1}}\gets y_{t_k}+y'_{t_k}(t_{k+1}-t_k)}
#' for \eqn{k=1, \dots, n-1}.
#'
#' @return A list of predicted values generated using Takens' theorem, based on the frequentist regularization regression with random features model,
#' specifying for time series data with the smooothness rate of change:
#'
#' \code{fit_results}: The result of fitted model from \code{glmnet::glmnet}.
#'
#' \code{W}: The matrix of random weights used of random features model.
#'
#' \code{b}: The vector of random biases used of random features model.
#'
#' \code{AIC}: The Akaike Information Criterion of the fitted model.
#'
#' \code{cv_model}: The result of cross-validation model from \code{cv.glmnet}.
#'
#' \code{y_pred}: A numeric vector of forecast values for future time steps, with length equal to \code{pred_size}.
#'
#' \code{pred.ci}: A matrix including the confidence interval using bootstrapping.
#'
#' \code{n_features}: A numeric value of the number of random features.
#'
#' \code{raw_diff}: A vector of the time derivatives.
#'
#' \code{smooth_diff}: A vector of the smooth time derivatives.
#'
#' @examples
#' \dontrun{
#' set.seed(456)
#' y = cumsum(rnorm(1000))
#' fitted_ts_freq_taken_reg_rfm <- ts_forecast_freq_reg_rfm_taken(
#'     ts_data=y, bootstrap_CI=TRUE,
#'     bootstrap_ref_point=900)
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
#'
ts_forecast_freq_reg_rfm_taken <- function(ts_data,
                                           time = NULL,
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
                                           alpha = NULL,
                                           lambda = c(10^-3, 10^-4, 10^-5, 10^-6, 10^-7),
                                           lambda_type = "min",
                                           standardize = TRUE,
                                           nfolds = 10,
                                           reg_type = "lasso",
                                           feature_selection = "sqrt",
                                           feature_constant = NULL,
                                           bootstrap_CI = FALSE,
                                           bootstrap_ref_point = NULL,
                                           CI = 95) {

  # Check input
  if (is.null(ts_data)) stop("Please check your input time series.")
  if (is.null(time)) time <- 1:length(ts_data)
  if (length(ts_data) != length(time)) stop("Time series must have the same length: ", length(ts_data), " vs ", length(time), ".")

  if (!(smooth_diff %in% c(TRUE, FALSE, T, F))) stop("`smooth_diff` must be a boolean value.")
  if (!(positive_int_check(window_size))) stop("`window_size` must be a positive integer.")
  if (!(positive_int_check(pred_size))) stop("`pred_size` must be a positive integer.")
  if (!(reg_type %in% c("lasso", "ridge"))) stop("`reg_type` is invalid. Please choose `lasso` or `ridge`.")

  if (window_size > length(ts_data)) stop("`window_size` exceeds the length of `ts_data`.")

  if (length(pred_time_step) == 1) pred_time_step <- rep(pred_time_step, pred_size)
  if (length(pred_time_step) != pred_size) stop("Please check your input of `pred_time_step`.")

  if (!(bootstrap_CI %in% c(TRUE, FALSE, T, F))) stop("`bootstrap_CI` must be a boolean value.")
  if (!(is.null(bootstrap_ref_point)) && !(positive_int_check(bootstrap_ref_point))){
    stop("`bootstrap_ref_point` must be a positive integer.")
  }

  # Currently support
  # *Error distribution: Gaussian distribution
  family = "gaussian"

  # *Finite difference: forward difference
  fdm = "forward"
  delta_t = diff(time, 1)

  # Compute rate of change
  if (fdm == "forward") raw_diff = na.omit(forward_difference(x=as.vector(ts_data), step=delta_t))

  if (smooth_diff %in% c(TRUE, T)){
    rate_of_change <- na.omit(smoothness_ts(ts_data=as.vector(raw_diff), fn_name=method, params=smooth_params))
    sdiff <- rate_of_change
  } else {
    rate_of_change <- raw_diff
    sdiff <- NULL
  }

  # Compute SE for CI
  if (bootstrap_CI %in% c(TRUE, T)){
    res_se <- se_ts_freq_taken_reg_rfm(
      ts_data = ts_data,
      time = time,
      ref_point = bootstrap_ref_point,
      smooth_diff = smooth_diff,
      method = method,
      smooth_params = smooth_params,
      window_size = window_size,
      pred_size = pred_size,
      pred_time_step = pred_time_step,
      weight_dist = weight_dist,
      weight_params = weight_params,
      bias_dist = bias_dist,
      bias_params = bias_params,
      act_func = act_func,
      alpha = alpha,
      lambda = lambda,
      lambda_type = lambda_type,
      standardize = standardize,
      nfolds = nfolds,
      reg_type = reg_type,
      feature_selection = feature_selection,
      feature_constant = feature_constant
    )
    se <- res_se$se
  } else {
    se <- NULL
  }

  train_dat_vec <- ts_data
  embed_dat_train <- embed(train_dat_vec, dimension = window_size + 1)

  if (nrow(embed_dat_train) > length(rate_of_change[window_size:length(rate_of_change)])) {
    stop("Please check the size of the input rate of change.")
  }
  x_train <- embed_dat_train[, 2:(window_size + 1)]

  # Reverse the data in time order
  x_train <- x_train[, window_size:1]

  # Choose appropriate index for the derivative values
  y_train <- rate_of_change[window_size:(nrow(x_train) + window_size - 1)]
  if (nrow(x_train) != length(y_train)) {
    stop("Please check dimensions of x_train and y_train")
  }
  # Reverse the data in time order
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
  AIC <- rfm_model$AIC

  x_prime_pred <- rep(NA, pred_size)
  x_prime_pred[1] <- predict.FreqRandFeatRegModel(rfm_model, x_test)

  y_pred <- rep(NA, pred_size)
  y_pred[1] <- taken_theorem_predict(
    rate_of_change = x_prime_pred[1],
    past_values = tail(train_dat_vec, 1),
    time_step = pred_time_step[1]
  )

  # Define the Reduce function to update predictions
  update_predictions <- function(acc, i) {
    fit_data <- c(train_dat_vec, acc$y_pred[1:(i - 1)])
    x_test <- matrix(
      tail(fit_data, window_size),
      nrow = 1,
      ncol = window_size,
      byrow = TRUE
    )

    pred <- predict.FreqRandFeatRegModel(rfm_model, x_test)

    acc$x_prime_pred[i] <- pred
    acc$y_pred[i] <- taken_theorem_predict(
      rate_of_change = pred,
      past_values = tail(fit_data, 1),
      time_step = pred_time_step[i]
    )

    return(acc)
  }

  # Use Reduce to iteratively update predictions
  results <- Reduce(
    update_predictions,
    2:pred_size,
    init = list(y_pred = y_pred, x_prime_pred = x_prime_pred)
  )

  alpha = (100-CI)/100
  critical_value <- qnorm((alpha/2)+(CI/100))
  if (bootstrap_CI %in% c(TRUE, T)){
    pred.ci <- upper_lower_ci(pred=results$y_pred, se=se, pred_size=pred_size, critical_value=critical_value)
  } else {
    pred.ci <- NULL
  }

  return(
    list(
      fit_results = fit_model,
      W = W,
      b = b,
      AIC = AIC,
      cv_model = cv_model,
      y_pred = results$y_pred,
      pred.ci = pred.ci,
      n_features = n_features,
      raw_diff = raw_diff,
      smooth_diff = sdiff
    )
  )
}
