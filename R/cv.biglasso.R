cv.biglasso <- function(X, y, row.idx = 1:nrow(X), ncores = 1, ...,
                        nfolds = 5, seed, cv.ind, trace = FALSE) {

  max.cores <- parallel::detectCores()
  if (ncores > max.cores) {
    ncores = max.cores
  }
  
  fit <- biglasso(X=X, y=y, row.idx = row.idx, ncores = ncores, ...)
  n <- fit$n
  E <- Y <- matrix(NA, nrow=n, ncol=length(fit$lambda))
  y <- fit$y

  if (fit$family == 'binomial') {
    PE <- E
  }
  
  if (!missing(seed)) set.seed(seed)
  if (missing(cv.ind)) {
    if (fit$family=="binomial" & (min(table(y)) > nfolds)) {
      ind1 <- which(y==1)
      ind0 <- which(y==0)
      n1 <- length(ind1)
      n0 <- length(ind0)
      cv.ind1 <- ceiling(sample(1:n1)/n1*nfolds)
      cv.ind0 <- ceiling(sample(1:n0)/n0*nfolds)
      cv.ind <- numeric(n)
      cv.ind[y==1] <- cv.ind1
      cv.ind[y==0] <- cv.ind0
    } else {
      cv.ind <- ceiling(sample(1:n)/n*nfolds)
    }
  }

  cv.args <- list(...)
  cv.args$lambda <- fit$lambda

  for (i in 1:nfolds) {
    if (trace) {
      cat("Starting CV fold #", i, "; time: ", format(Sys.time()), sep="","\n")
    }
    res <- cvf(i, X, y, cv.ind, cv.args)
    E[cv.ind==i, 1:res$nl] <- res$loss
    if (fit$family=="binomial") PE[cv.ind==i, 1:res$nl] <- res$pe
    Y[cv.ind==i, 1:res$nl] <- res$yhat
  }

  ## Eliminate saturated lambda values, if any
  ind <- which(apply(is.finite(E), 2, all))
  E <- E[,ind]
  Y <- Y[,ind]
  lambda <- fit$lambda[ind]

  ## Return
  cve <- apply(E, 2, mean)
  cvse <- apply(E, 2, sd) / sqrt(n)
  min <- which.min(cve)

  val <- list(cve = cve, cvse = cvse, lambda = lambda, fit = fit, min = min, 
              lambda.min=lambda[min], 
              null.dev = mean(loss.biglasso(y, rep(mean(y), n), fit$family)))
  if (fit$family=="binomial") {
    pe <- apply(PE, 2, mean)
    val$pe <- pe[is.finite(pe)]
  }
  structure(val, class=c("cv.biglasso", "cv.ncvreg"))
}

cvf <- function(i, XX, y, cv.ind, cv.args) {
  cv.args$X <- XX
  cv.args$y <- y
  cv.args$row.idx <- which(cv.ind != i)
  cv.args$warn <- FALSE
  idx.test <- which(cv.ind == i)
  fit.i <- do.call("biglasso", cv.args)

  y2 <- y[cv.ind==i]
  yhat <- matrix(predict(fit.i, XX, row.idx = idx.test, type="response"), length(y2))
  loss <- loss.biglasso(y2, yhat, fit.i$family)
  pe <- if (fit.i$family=="binomial") {(yhat < 0.5) == y2} else NULL
  list(loss=loss, pe=pe, nl=length(fit.i$lambda), yhat=yhat)
}
