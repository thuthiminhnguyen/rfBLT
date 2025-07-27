#' Confidence Interval Construction for Time series predictions
#'
#' This function helps to produce the confidence interval by inputting the predicted values, the standard errors, the prediction time steps and the critical value.
#'
#' @param pred A numeric vector of the predictions.
#' @param se A numeric vector of the standard errors corresponding to the prediction time steps.
#' @param pred_size A positive integer of the prediction time steps.
#' @param critical_value A numeric value of the critical value for constructing confidence intervals. Defaults to `qnorm(0.975)`
#'
#' @return A list contains the lower and upper bounds of the confidence intervals:
#'
#' `lower`: The lower bound of the confidence intervals.
#'
#' `upper`: The upper bound of the confidence intervals.
#'
#' @import stats
#'
upper_lower_ci <- function(pred, se, pred_size, critical_value=qnorm(0.975)) {
  se <- as.vector(se)
  pred <- as.vector(pred)
  if (length(se) != pred_size | length(pred) != length(se)) {
    stop("Difference between size of se and pred_size.")
  }

  upper <- pred + critical_value * se
  lower <- pred - critical_value * se

  return(list(lower = lower, upper = upper))
}

#' Calculate standard error for confidence interval of time series data when fitting frequentist regularization with random features model using bootstrapping
#'
#' This function aims to compute the standard error of model prediction given the time series data.
#'
#' @param ts_data A numeric vector representing the time series data.
#' @param ref_point An integer of the starting training point for calculating the standard error.
#'   Defaults to `length(ts_data)/3`if it is greater than `pred_size`, otherwise it is `length(ts_data)-pred_size`.
#' @param window_size A positive integer of the embedding dimension of time series. Defaults to `9`.
#' @param pred_size A positive integer of the prediction time steps. Defaults to `7`.
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
#' @param standardize A logical flag to standardize the transformation of the random features model for `glmnet::glmnet()`. Defaults to `TRUE`.
#' @param nfolds An integer of the number of folds. Defaults to `10`.
#' @param reg_type A character string of the regularization method.
#'  Possible values: `"lasso"` (default), `"ridge"`, `"elastic net"`.
#' @param alpha A numeric value for the elastic net mixing parameter.
#'   Use `alpha = 1` for LASSO, `alpha = 0` for Ridge, `alpha = 0.5` (default) for Elastic net.
#'   If `reg_type` is `"lasso"` or `"ridge"`, `alpha` can be set to `NULL`.
#'   Defaults to `NULL`.
#' @param lambda A numeric vector of possible lambda values to use in cross-validation. Defaults to `c(10^-3, 10^-4, 10^-5, 10^-6, 10^-7)`.
#' @param lambda_type A character string indicating the type of lambda to use after cross-validation.
#'   Possible values are `"min"` (default), and `"1se"`.
#' @param feature_selection A character string specifying the method for selecting the number of random features.
#'   Possible values are:
#'   - `"sqrt"` (default): The number of features is set to the square root of the number of rows of the training matrix `nrow`, i.e., `sqrt(nrow)`.
#'   - `"factor"`: The number of features is `feature_constant*nrow`.
#'   - `"constant"`: The number of features is fixed at `feature_constant`.
#' @param feature_constant A numeric value used when `feature_selection` is `"factor"` or `"constant"`.
#'   Ignored when `feature_selection = "sqrt"`. Defaults to `NULL`.
#'
#' @return A list containing standard error and relevant components:
#' \itemize{
#'   \item \code{se}: The standard error of the predictions.
#'   \item \code{pred_mat}: A matrix containing predictions, where the columns represent the prediction size.
#'   \item \code{mat_true}: A matrix containing true values of the input time series, where the columns represent the prediction size.
#' }
#'
#' @examples
#' \dontrun{
#' set.seed(456)
#' y = cumsum(rnorm(1000))
#' res_se <- se_ts_freq_reg_rfm(ts_data = y, ref_point = 900)
#' }
#'
#' @import glmnet
se_ts_freq_reg_rfm <- function(ts_data,
                               ref_point = NULL,
                               window_size = 9,
                               pred_size = 7,
                               weight_dist = "normal",
                               weight_params = list(),
                               bias_dist = "uniform",
                               bias_params = list(min_val = 0, max_val = 2*pi),
                               act_func = "fourier",
                               standardize = TRUE,
                               nfolds = 10,
                               reg_type = "lasso",
                               alpha = NULL,
                               lambda = c(10^-3, 10^-4, 10^-5, 10^-6, 10^-7),
                               lambda_type = "min",
                               feature_selection = "sqrt",
                               feature_constant = NULL) {
  # Validate parameters
  m = length(ts_data)
  if (is.null(ref_point)) {
    if (floor(m/3) > pred_size) {
      ref_point <- floor(2*m/3)
    } else {
      ref_point <- m-pred_size
    }
  }
  family = "gaussian"

  if (!positive_int_check(ref_point)) stop("`ref_point` must be a positive integer.")
  if (ref_point + pred_size > m) {
    stop("`ref_point + pred_size` must be less than or equal to `length(ts_data)`. Adjust the parameters.")
  }
  if (ref_point + pred_size > length(ts_data)) {
    stop("`ref_point+pred_size` must not exceed `length(ts_data)`. Adjust `ref_point` or `pred_size`.")
  }
  if (ref_point < window_size + 2) {
    stop("Invalid `ref_point`.")
  }

  size_se <- m - ref_point - pred_size + 1 # size for computing the standard errors

  # Vectorized operations for prediction matrix
  fit_data_list <- lapply(1:size_se, function(i)
    ts_data[1:(ref_point + i - 1)])
  pred_res_list <- lapply(fit_data_list, function(fit_data) {
    ts_forecast_freq_reg_rfm(
      ts_data = fit_data,
      window_size = window_size,
      pred_size = pred_size,
      weight_dist = weight_dist,
      weight_params = weight_params,
      bias_dist = bias_dist,
      bias_params = bias_params,
      act_func = act_func,
      standardize = standardize,
      nfolds = nfolds,
      alpha = alpha,
      lambda = lambda,
      lambda_type = lambda_type,
      reg_type = reg_type,
      feature_selection = feature_selection,
      feature_constant = feature_constant,
      bootstrap_CI = FALSE
    )
  })
  pred_mat <- do.call(rbind, lapply(pred_res_list, function(pred_res)
    pred_res$y_pred))

  # Vectorized operations for true matrix
  mat_true <- t(sapply(1:size_se, function(i) {
    start_index <- ref_point + i
    end_index <- start_index + pred_size - 1
    ts_data[start_index:end_index]
  }))

  # Compute the standard errors
  se <- sqrt(colMeans((pred_mat - mat_true)^2))

  return(list(
    se = se,
    pred_mat = pred_mat,
    mat_true = mat_true
  ))
}

#' Calculate standard error for confidence interval of time series data when fitting frequentist regularization regression with random features for time delay embedding using bootstrapping
#'
#' This function aims to compute the standard error of model prediction given the time series data using bootstrapping.
#'
#' @param ts_data A numeric vector representing the time series data.
#' @param time A numeric vector representing the time gap between two consecutive observations in time series. Defaults to `NULL`.
#' @param ref_point An integer of the starting training point for calculating the standard error.
#'   Defaults to `length(ts_data)/3`if it is greater than `pred_size`, otherwise it is `length(ts_data)-pred_size`.
#' @param smooth_diff A boolean value  to choose to smooth the time derivatives.
#'  Possible values: `TRUE` (default), `T`, `FALSE`, `F`.
#' @param method A character string of the smooth method for the time derivatives.
#'  Defaults to `ma`. Other supported options include `"polynomial"`, `"localized_polynomial"`, `"spline"`.
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
#' @param window_size An integer of the embedding dimension of time series. Defaults to `9`.
#' @param pred_size An integer of the prediction time steps. Defaults to `7`.
#' @param pred_time_step An integer or a vector of time gap between two consecutive observations. Defaults to `1`.
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
#' @param alpha A numeric value for the elastic net mixing parameter.
#'   Use `alpha = 1` for LASSO, `alpha = 0` for Ridge, `alpha = 0.5` (default) for Elastic net.
#'   If `reg_type` is `"lasso"` or `"ridge"`, `alpha` can be set to `NULL`.
#'   Defaults to `NULL`.
#' @param lambda A numeric vector of possible lambda values to use in cross-validation. Defaults to `c(10^-3, 10^-4, 10^-5, 10^-6, 10^-7)`.
#' @param lambda_type A character string indicating the type of lambda to use after cross-validation.
#'   Possible values are `"min"` (default), and `"1se"`.
#' @param standardize A logical flag to standardize the transformation of the random features model. Defaults to `TRUE`.
#' @param nfolds An integer of the number of folds. Defaults to `10`.
#' @param reg_type A character string of the regularization method.
#'  Possible values: `"lasso"` (default), `"ridge"`, `"elastic net"`.
#' @param feature_selection A character string specifying the method for selecting the number of random features.
#'   Possible values are:
#'   - `"sqrt"` (default): The number of features is set to the square root of the number of rows of the training matrix `nrow`, i.e., `sqrt(nrow)`.
#'   - `"factor"`: The number of features is `feature_constant*nrow`.
#'   - `"constant"`: The number of features is fixed at `feature_constant`.
#' @param feature_constant A numeric value used when `feature_selection` is `"factor"` or `"constant"`.
#'   Ignored when `feature_selection = "sqrt"`. Defaults to `NULL`.
#'
#' @return A list containing standard error and relevant components:
#' \itemize{
#'   \item \code{se}: The standard error of the predictions.
#'   \item \code{pred_mat}: A matrix containing predictions, where the columns represent the prediction size.
#'   \item \code{mat_true}: A matrix containing true values of the input time series, where the columns represent the prediction size.
#' }
#'
#' @examples
#' \dontrun{
#' set.seed(456)
#' y = cumsum(rnorm(1000))
#' res_se <- se_ts_freq_taken_reg_rfm(ts_data = y, ref_point = 900)
#' }
#'
#' @import glmnet
se_ts_freq_taken_reg_rfm <- function(ts_data,
                                     time,
                                     ref_point = NULL,
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
                                     feature_constant = NULL) {
  # Validate parameters
  m = length(ts_data)
  if (is.null(ref_point)) {
    if (floor(m/3) > pred_size) {
      ref_point <- floor(2*m/3)
    } else {
      ref_point <- m-pred_size
    }
  }
  family = "gaussian"

  if (!positive_int_check(ref_point)) stop("`ref_point` must be a positive integer.")
  if (ref_point + pred_size > length(ts_data)) {
    stop("`ref_point + pred_size` must not exceed the length of `ts_data`. Adjust `ref_point` or `pred_size`.")
  }
  if (ref_point < window_size + 2) {
    stop("Invalid `ref_point`")
  }

  size_se <- m - ref_point - pred_size + 1 # size for computing the standard errors

  # Vectorized operations for prediction matrix
  fit_data_list <- lapply(1:size_se, function(i) list(
    fit_data = ts_data[1:(ref_point + i - 1)],
    fit_time = time[1:(ref_point + i - 1)]
  ))

  pred_res_list <- lapply(fit_data_list, function(fit_data_pair) {
    ts_forecast_freq_reg_rfm_taken(
      ts_data = fit_data_pair$fit_data,
      time = fit_data_pair$fit_time,
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
      feature_constant = feature_constant,
      bootstrap_CI = FALSE
    )
  })

  pred_mat <- do.call(rbind, lapply(pred_res_list, function(pred_res) pred_res$y_pred))

  # Vectorized operations for true matrix
  mat_true <- t(sapply(1:size_se, function(i) {
    start_index <- ref_point + i
    end_index <- start_index + pred_size - 1
    ts_data[start_index:end_index]
  }))

  # Compute the standard errors, when n is large
  se <- sqrt(colMeans((pred_mat - mat_true)^2))

  return(list(se = se, pred_mat = pred_mat, mat_true = mat_true))
}

# --- CREDIBLE INTERVAL ---
credible_interval <- function(x, alpha = 0.05){
  if (!is.null(ncol(x))){
    ci = lapply(1:ncol(x), function(i) stats::quantile(x[,i], probs=c(alpha/2, (1-alpha/2))))
  } else {
    ci = quantile(x, probs=c(alpha/2, (1-alpha/2)))
  }
  # Convert the list of CIs to a matrix
  ci_matrix <- do.call(rbind, ci)
  colnames(ci_matrix) <- c("lower.CI", "upper.CI")

  return(ci_matrix)
}
