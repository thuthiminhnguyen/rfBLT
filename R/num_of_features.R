# Number of feature
check_num_feature <- function(feature_selection, feature_constant, n){
  if (!(feature_selection %in% c("sqrt", "factor", "constant"))){
    stop("Please choose the method of feature selection.")
  } else if (feature_selection == "sqrt" && !is.null(feature_constant)){
    warning("feature_constant will not be considered.")
  }
  # else if (((feature_selection == "sqrt") && (sqrt(n) %% 1 != 0)) |
  #            ((feature_selection == "factor") && (feature_constant * n %% 1 != 0)) |
  #            ((feature_selection == "constant") && (feature_constant %% 1 != 0))){
  #   warning("The number of feature will be rounded.")
  # }
  n_features <- switch(
    feature_selection,
    "sqrt" = floor(sqrt(n)),
    "factor" = floor(n * feature_constant),
    "constant" = floor(feature_constant),
    stop("Invalid feature selection method. Please choose from 'sqrt', 'factor', 'constant'.")
  )
  return(n_features)
}
