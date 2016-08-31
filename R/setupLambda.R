setupLambda <- function(X, y, row.idx, center, scale, family, alpha, 
                           lambda.min, nlambda, penalty.factor) {
  n <- length(row.idx)
  p <- ncol(X)

  ## Determine lambda.max
  col.idx <- which(penalty.factor!=0)
  if (length(col.idx)!=p) {
    stop("Current version doesn't allow unpenalized covariates for big.matrix!")
    # fit <- glm(y~X[, -ind], family=family)
  } else {
    fit <- glm(y~1, family=family)
  }
 
  # cat("Compute lambda_max, start: ", format(Sys.time()), "\n")
  if (family=="gaussian") {
    zmax <- .Call("maxprod_bm", X@address, fit$residuals, row.idx, center, scale, 
                  col.idx, penalty.factor) / n
  } else {
    zmax <- .Call("maxprod_bm", X@address, residuals(fit, "working") * fit$weights, 
                  row.idx, center, scale, col.idx, penalty.factor) / n
  }
  lambda.max <- zmax/alpha
  # cat("Compute lambda_max, end: ", format(Sys.time()), "\n")

  if (lambda.min==0) {
    lambda <- c(exp(seq(log(lambda.max),log(.001*lambda.max),len=nlambda-1)),0)
  } else {
    lambda <- exp(seq(log(lambda.max),log(lambda.min*lambda.max),len=nlambda))
  }
  lambda
}
