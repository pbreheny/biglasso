biglasso <- function(X, y, row.idx = 1:nrow(X), penalty = c("lasso", "ridge", "enet"),
                     family = c("gaussian","binomial"), alpha = 1, 
                     lambda.min = ifelse(nrow(X) > ncol(X),.001,.05), nlambda = 100,
                     lambda, eps = .001, max.iter = 1000, dfmax = ncol(X)+1, 
                     penalty.factor = rep(1, ncol(X)), warn = TRUE) {
  # Coersion
  family <- match.arg(family)
  penalty <- match.arg(penalty)
  
  if (identical(penalty, "lasso")) {
    alpha <- 1
  } else if (identical(penalty, 'ridge')) {
    alpha <- 0.0001 ## equivalent to ridge regression
  } else {
    if (alpha >= 1 || alpha <= 0) {
      stop("alpha must be between 0 and 1 for elastic net penalty.")
    } else {
      alpha <- alpha
    }
  }
 
  if (nlambda < 2) stop("nlambda must be at least 2")

  # subset of the response vector
  y <- y[row.idx]
  
  if (any(is.na(y))) stop("Missing data (NA's) detected.  Take actions (e.g., removing cases, removing features, imputation) to eliminate missing data before fitting the model.")
  
  if (class(y) != "numeric") {
    tmp <- try(y <- as.numeric(y), silent=TRUE)
    if (class(tmp)[1] == "try-error") stop("y must numeric or able to be coerced to numeric")
  }
  
  if (family == 'binomial') {
    if (length(table(y)) > 2) {
      stop("Attemping to use family='binomial' with non-binary data")
    }
    if (!identical(sort(unique(y)), 0:1)) {
      y <- as.numeric(y==max(y))
    }
  }
  
  if (family=="gaussian") {
    yy <- y - mean(y)
  } else {
    yy <- y
  }
  
  ## TODO:
  ## assume inherits(X, "big.matrix"), AND no missing values
  p <- ncol(X)
  if (length(penalty.factor) != p) stop("penalty.factor does not match up with X")
  if (storage.mode(penalty.factor) != "double") storage.mode(penalty.factor) <- "double"

  n <- length(row.idx) ## subset of X. idx: indices of rows.

  # standardize X, return center vector and scale vector
  stand <- .Call('standardize_bm', X@address, as.integer(row.idx-1), PACKAGE = 'biglasso')
  center <- stand[[1]]
  scale <- stand[[2]]
  
  ## TODO: eliminate variables with variance close to 0.
  nz <- 1:p
#   nz <- which(scale > 1e-6)
#   if (length(nz) != ncol(XX)) XX <- XX[ ,nz, drop=FALSE]
#   penalty.factor <- penalty.factor[nz]
  
  if (missing(lambda)) {
    lambda <- setupLambda(X, yy, as.integer(row.idx-1), center, scale, family, alpha, lambda.min, nlambda, penalty.factor)
    user.lambda <- FALSE
  } else {
    nlambda <- length(lambda)
    user.lambda <- TRUE
  }

  ## fit model
  if (family == 'gaussian') {
    res <- .Call("cdfit_gaussian", X@address, yy, as.integer(row.idx-1), 
                 center, scale, lambda, eps, as.integer(max.iter), penalty.factor, 
                 alpha, as.integer(dfmax), 
                 as.integer(user.lambda | any(penalty.factor==0)),
                 PACKAGE = 'biglasso')

    a <- rep(mean(y), nlambda)
    b <- Matrix(res[[1]], p, nlambda, sparse = T)
    loss <- res[[2]]
    iter <- res[[3]]
  } else if (family == 'binomial') {
    res <- .Call("cdfit_binomial", X@address, yy, as.integer(row.idx-1),
                 center, scale, lambda, eps, as.integer(max.iter), penalty.factor,
                 alpha, as.integer(dfmax),
                 as.integer(user.lambda | any(penalty.factor==0)), as.integer(warn),
                 PACKAGE = 'biglasso')
    a <- res[[1]]
    b <- Matrix(res[[2]], p, nlambda, sparse = T)
    loss <- res[[3]]
    iter <- res[[4]]
  } else {
    stop("Current version only supports Gaussian or Binominal response!")
  }
 
  ## Eliminate saturated lambda values, if any
  ind <- !is.na(iter)
  if (family != "gaussian") a <- a[ind]
  b <- b[, ind, drop=FALSE]
  iter <- iter[ind]
  lambda <- lambda[ind]
  loss <- loss[ind]
  
  if (warn & any(iter==max.iter)) warning("Algorithm failed to converge for some values of lambda")
  
  ## Unstandardize coefficients.
  beta <- Matrix(0, nrow=(p+1), ncol=length(lambda), sparse = T)
  bb <- b/scale[nz]
  beta[nz+1,] <- bb
  beta[1,] <- a - crossprod(center[nz], bb)

  ## Names
  varnames <- if (is.null(colnames(X))) paste("V", 1:p, sep="") else colnames(X)
  if (family!="gaussian") varnames <- c("(Intercept)", varnames)
  dimnames(beta) <- list(varnames, round(lambda,digits=4))

  ## Output
  val <- structure(list(beta = beta,
                        iter = iter,
                        lambda = lambda,
                        penalty = penalty,
                        family = family,
                        alpha = alpha,
                        loss = loss,
                        penalty.factor = penalty.factor,
                        n = n,
                        center = center,
                        scale = scale,
                        y = y),
                   class = c("biglasso", 'ncvreg'))
  val
}
