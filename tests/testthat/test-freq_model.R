test_that("fit_freq_model returns correct structure", {
  set.seed(123)
  x <- matrix(rnorm(100 * 5), ncol = 5)
  y <- rnorm(100)

  model <- fit_freq_reg_rfm(x, y, n_features = 50)

  expect_s3_class(model, "FreqRandFeatRegModel")
  expect_true(!is.null(model$fit_results))
  expect_true(!is.null(model$rf_generator))
  expect_equal(length(model$train_preds), nrow(x))
})

test_that("predict works correctly", {
  set.seed(123)
  x <- matrix(rnorm(200 * 5), ncol = 5)
  y <- rnorm(200)
  newx <- matrix(rnorm(10 * 5), ncol = 5)

  model <- fit_freq_reg_rfm(x, y)
  preds <- predict.FreqRandFeatRegModel(model, newx)

  expect_type(preds, "double")
  expect_equal(length(preds), nrow(newx))
})
