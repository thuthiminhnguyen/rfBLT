# Activation functions
sigmoid <- function(x) {
  return(1 / (1 + exp(-x)))
}

relu <- function(x) {
  return(pmax(0, x))
}

tanh_activation <- function(x) {
  return(tanh(x))
}

softmax <- function(x) {
  if (is.matrix(x)) {
    # For matrices, apply softmax row-wise
    row_max <- apply(x, 1, max)
    exp_x <- exp(sweep(x, 1, row_max, "-"))
    return(sweep(exp_x, 1, rowSums(exp_x), "/"))
  } else {
    # For vectors, apply regular softmax
    exp_x <- exp(x - max(x))
    return(exp_x / sum(exp_x))
  }
}

sin_activation <- function(x) {
  return(sin(x))
}

cos_activation <- function(x) {
  return(cos(x))
}

# Random Fourier Features
fourier_activation <- function(x) {
  if (!is.null(nrow(x)) | !is.null(ncol(x))){
    D <- ncol(x)
  } else {
    D <- 1
  }
  return(sqrt(2/D)*cos(x))
}

# Wrapper for activation functions
activation_function <- function(x, act_func="relu") {

  Z <- switch(act_func,
              "sigmoid" = sigmoid(x),
              "relu" = relu(x),
              "tanh" = tanh_activation(x),
              # "softmax" = softmax(x),
              "sine" = sin_activation(x),
              "cosine" = cos_activation(x),
              "fourier" = fourier_activation(x),
              stop("Invalid activation function. Choose from: sigmoid, relu, tanh, sine, cosine, fourier"))

  if (!is.null(nrow(x)) | !is.null(ncol(x))){
    Z <- matrix(Z, nrow = nrow(x), ncol = ncol(x))
  } else {
    Z <- as.vector(Z)
  }

  return(Z)
}
