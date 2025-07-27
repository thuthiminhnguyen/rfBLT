# Validation of input for random feature model
validate_params <- function(dist, params, name) {
  if (length(params) > 0 && is.null(names(params))) {
    stop(paste0("Unused arguments in `", name, "`"))
  }

  if (!is.list(params)) {
    stop(paste0("`", name, "` must be a list."))
  }

  if (dist == "normal") {
    required_keys <- c("mean", "sd")
    missing_keys <- setdiff(required_keys, names(params))
    unused_keys  <- setdiff(names(params), required_keys)

    if (length(unused_keys) > 0) {
      stop(paste0("Unused parameters in `", name, "`: ", paste(unused_keys, collapse = ", ")))
    }

  } else if (dist == "uniform") {
    required_keys <- c("min_val", "max_val")
    missing_keys <- setdiff(required_keys, names(params))
    unused_keys  <- setdiff(names(params), required_keys)

    if (length(unused_keys) > 0) {
      stop(paste0("Unused parameters in `", name, "`: ", paste(unused_keys, collapse = ", ")))
    }

  } else if (dist == "cauchy") {
    required_keys <- c("location", "scale")
    missing_keys <- setdiff(required_keys, names(params))
    unused_keys  <- setdiff(names(params), required_keys)

    if (length(unused_keys) > 0) {
      stop(paste0("Unused parameters in `", name, "`: ", paste(unused_keys, collapse = ", ")))
    }

  } else if (dist == "exponential"){
    required_keys <- c("rate")
    missing_keys <- setdiff(required_keys, names(params))
    unused_keys  <- setdiff(names(params), required_keys)

    if (length(unused_keys) > 0) {
      stop(paste0("Unused parameters in `", name, "`: ", paste(unused_keys, collapse = ", ")))
    }

  } else if (dist == "bernoulli"){
    required_keys <- c("prob")
    missing_keys <- setdiff(required_keys, names(params))
    unused_keys  <- setdiff(names(params), required_keys)

    if (length(unused_keys) > 0) {
      stop(paste0("Unused parameters in `", name, "`: ", paste(unused_keys, collapse = ", ")))
    }

  } else if (dist == "lognormal"){
    required_keys <- c("meanlog", "sdlog")
    missing_keys <- setdiff(required_keys, names(params))
    unused_keys  <- setdiff(names(params), required_keys)

    if (length(unused_keys) > 0) {
      stop(paste0("Unused parameters in `", name, "`: ", paste(unused_keys, collapse = ", ")))
    }

  }
}

# Check positive integer
positive_int_check <- function(x) {
  return(x%%1 == 0 && x > 0)
}
