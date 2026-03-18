
<!-- README.md is generated from README.Rmd. Please edit that file -->

# rfBLT: Random Feature Bayesian Lasso Takens

<!-- badges: start -->

<!-- badges: end -->

An R package for implementing Regularized or Sparse Random Feature
Models integrating with Time Delay Embedding Theorem, within both
Frequentist and Bayesian frameworks. Currently, only Gaussian error
models are supported. In the Frequentist setting, lasso, ridge, and
elastic net regularization are available, while the Bayesian setting
provides shrinkage priors for lasso and ridge. Recursive prediction is
implemented for time series forecasting. The package also supports
cross-sectional data and time series prediction using lagged features,
without Takens’ theorem, through standard Sparse Random Feature Models.

## Installation

You can install the development version of rfBLT from
[GitHub](https://github.com/thuthiminhnguyen/rfBLT) with:

``` r
# install.packages("devtools")
devtools::install_github("thuthiminhnguyen/rfBLT")
```

The package now can be loaded and used:

``` r
library(rfBLT)
```

## Example

The following example illustrates how to fit and forecast time series
values using the Sparse Random Feature Model using Takens’ Theorem,
within both Bayesian and Frequentist approaches.

``` r
# Generate data
set.seed(456)
y = cumsum(rnorm(1000))

# Visualize time series
plot(y, type="l", main="Time series data", xlab="Time index", ylab="Value")
```

### Bayesian

The `ts_forecast_bayes_reg_rfm_taken` function fits a Bayesian model to
a time series and forecasts future values for a specified prediction
horizon and confidence level. The example below demonstrates the model
with a smoothing method applied:

``` r
fit_bayes_reg_rfm_taken <- ts_forecast_bayes_reg_rfm_taken(ts_data=y, pred_size=7, CI=95, smooth_diff = TRUE)
```

The element `ts_forecast_bayes_reg_rfm_taken$fit_results` returns an
object of class bayesreg, which can be summarized as follows:

``` r
summary(fit_bayes_reg_rfm_taken$fit_results)
```

The output includes both smoothed and non-smoothed time derivatives for
diagnostic purposes.

``` r
# Plot the smoothed and non-smoothed time derivatives
plot(fit_bayes_reg_rfm_taken$raw_diff, type="l", 
     main="Smoothed and non-smoothed time derivatives",
     ylab="Time derivatives", xlab="Time index")
lines(fit_bayes_reg_rfm_taken$smooth_diff, col="blue", lwd=2)
legend("topleft",                
       legend = c("Non-smoothed", "Smoothed"),  
       col = c("black", "blue"), 
       lty = 1, lwd = c(1, 2), bty = "n") 
```

A plot function is implemented to plot Effective Sample Size
`plot.ess_plot()`.

``` r
plot(fit_bayes_reg_rfm_taken$ess)
```

The mean predicted values obtained from the model are

``` r
fit_bayes_reg_rfm_taken$y_pred
```

The 95% credible intervals for the forecasts are:

``` r
fit_bayes_reg_rfm_taken$pred.ci
```

### Frequentist

The following code demonstrates a Frequentist Sparse Random Feature
Model with time-delay embedding, without bootstrapped confidence
intervals:

``` r
fit_freq_reg_rfm_taken <- ts_forecast_freq_reg_rfm_taken(ts_data=y, pred_size=7, CI=95, smooth_diff = TRUE)
```

The element `fit_freq_reg_rfm_taken$fit_results` returns an object of
class glmnet, and its coefficients can be summarized as follows:

``` r
coef(fit_freq_reg_rfm_taken$fit_results)
```

Similar to the Bayesian approach, both smoothed and non-smoothed time
derivatives are computed and can be visualized.

``` r
# Plot the smoothed and non-smoothed time derivatives
plot(fit_freq_reg_rfm_taken$raw_diff, type="l", 
     main="Smoothed and non-smoothed time derivatives",
     ylab="Time derivatives", xlab="Time index")
lines(fit_freq_reg_rfm_taken$smooth_diff, col="blue", lwd=2)
legend("topleft",                
       legend = c("Non-smoothed", "Smoothed"),  
       col = c("black", "blue"), 
       lty = 1, lwd = c(1, 2), bty = "n") 
```

The prediction values and their confidence intervals (if applicable)
are:

``` r
# Predictions
fit_freq_reg_rfm_taken$y_pred

# Confidence interval
fit_freq_reg_rfm_taken$pred.ci
```

## Numerical experiments

For detailed examples and performance evaluations on real datasets
including COVID-19 data in Canada, and the S&P500 index, please refer to
our repository
[link](https://github.com/thuthiminhnguyen/rfBLT-numerical-experiments).

## References

\[1\] “Makalic E, Schmidt D (2016). "High-Dimensional Bayesian
Regularisedwith the Bayesreg Package." arXiv:1611.06649.”  
\[2\] “Friedman J, Hastie T, Tibshirani R (2010). "Regularization Paths
forLinear Models via Coordinate Descent." *Journal ofSoftware*, *33*(1),
1-22. <a href="doi:10.18637/jss.v033.i01\n"
class="uri">doi:10.18637/jss.v033.i01\n</a><https://doi.org/10.18637/jss.v033.i01>.”
