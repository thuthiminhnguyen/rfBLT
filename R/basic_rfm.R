# Random Feature Model
# Use W, b from train set for test set
transformation_and_activation <- function(x, W, b, act_func) {
  # Linear combination
  # x: n x p
  # W: p x n_features
  # b: n x n_features, with the number of different values are at most n_features

  # Check matrix input
  if (!is.matrix(x)){
    stop("x must be a matrix.")
  }
  if (!is.matrix(W)){
    stop("W must be a matrix.")
  }
  trans <- x%*%W+matrix(b, nrow=nrow(x), ncol=ncol(W), byrow=TRUE)

  # Activation function
  Z <- activation_function(trans, act_func=act_func)
  return(Z)
}

# Random Feature Model - phi(W^T X + b)
rfm <- function(x, n_features=100,
                weight_dist="normal", weight_params=list(),
                bias_dist="uniform", bias_params=list(min_val=0, max_val=2*pi),
                act_func="relu") {

  if (is.null(nrow(x)) | is.null(ncol(x))){
    stop("Input must be a matrix.")
  }

  # Number of dimensions
  d <- ncol(x)

  # Generate random weight matrix based on a specific distribution
  W <- generate_random_weights(input_dim=d, output_dim=n_features, weight_dist=weight_dist, weight_params=weight_params)

  # Generate random biases based on a specific distribution
  b <- generate_bias(n=n_features, bias_dist=bias_dist, bias_params=bias_params)

  # Apply activation function
  Z <- transformation_and_activation(x, W, b, act_func)

  # Return Z, W, and b for verification
  return(list(Z=Z, W=W, b=b))
}
