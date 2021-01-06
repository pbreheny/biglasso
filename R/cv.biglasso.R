#' Cross-validation for biglasso
#' 
#' Perform k-fold cross validation for penalized regression models over a grid
#' of values for the regularization parameter lambda.
#' 
#' The function calls \code{biglasso} \code{nfolds} times, each time leaving
#' out 1/\code{nfolds} of the data.  The cross-validation error is based on the
#' residual sum of squares when \code{family="gaussian"} and the binomial
#' deviance when \code{family="binomial"}.\cr \cr The S3 class object
#' \code{cv.biglasso} inherits class \code{\link[ncvreg]{cv.ncvreg}}.  So S3
#' functions such as \code{"summary", "plot"} can be directly applied to the
#' \code{cv.biglasso} object.
#' 
#' @param X The design matrix, without an intercept, as in
#' \code{\link{biglasso}}.
#' @param y The response vector, as in \code{biglasso}.
#' @param row.idx The integer vector of row indices of \code{X} that used for
#' fitting the model. as in \code{biglasso}.
#' @param eval.metric The evaluation metric for the cross-validated error and
#' for choosing optimal \code{lambda}. "default" for linear regression is MSE
#' (mean squared error), for logistic regression is misclassification error.
#' "MAPE", for linear regression only, is the Mean Absolute Percentage Error.
#' @param ncores The number of cores to use for parallel execution across a
#' cluster created by the \code{parallel} package. (This is different from
#' \code{ncores} in \code{\link{biglasso}}, which is the number of OpenMP
#' threads.)
#' @param ... Additional arguments to \code{biglasso}.
#' @param nfolds The number of cross-validation folds.  Default is 5.
#' @param seed The seed of the random number generator in order to obtain
#' reproducible results.
#' @param cv.ind Which fold each observation belongs to.  By default the
#' observations are randomly assigned by \code{cv.biglasso}.
#' @param trace If set to TRUE, cv.biglasso will inform the user of its
#' progress by announcing the beginning of each CV fold.  Default is FALSE.
#' @return An object with S3 class \code{"cv.biglasso"} which inherits from
#' class \code{"cv.ncvreg"}.  The following variables are contained in the
#' class (adopted from \code{\link[ncvreg]{cv.ncvreg}}).  \item{cve}{The error
#' for each value of \code{lambda}, averaged across the cross-validation
#' folds.} \item{cvse}{The estimated standard error associated with each value
#' of for \code{cve}.} \item{lambda}{The sequence of regularization parameter
#' values along which the cross-validation error was calculated.}
#' \item{fit}{The fitted \code{biglasso} object for the whole data.}
#' \item{min}{The index of \code{lambda} corresponding to \code{lambda.min}.}
#' \item{lambda.min}{The value of \code{lambda} with the minimum
#' cross-validation error.} \item{null.dev}{The deviance for the intercept-only
#' model.} \item{pe}{If \code{family="binomial"}, the cross-validation
#' prediction error for each value of \code{lambda}.} \item{cv.ind}{Same as
#' above.}
#' @author Yaohui Zeng and Patrick Breheny
#' 
#' Maintainer: Yaohui Zeng <yaohui.zeng@@gmail.com>
#' @seealso \code{\link{biglasso}}, \code{\link{plot.cv.biglasso}},
#' \code{\link{summary.cv.biglasso}}, \code{\link{setupX}}
#' @examples
#' \dontrun{
#' ## cv.biglasso
#' data(colon)
#' X <- colon$X
#' y <- colon$y
#' X.bm <- as.big.matrix(X)
#' 
#' ## logistic regression
#' cvfit <- cv.biglasso(X.bm, y, family = 'binomial', seed = 1234, ncores = 2)
#' par(mfrow = c(2, 2))
#' plot(cvfit, type = 'all')
#' summary(cvfit)
#' }
#' 
#' @export cv.biglasso
#' 
cv.biglasso <- function(X, y, row.idx = 1:nrow(X), eval.metric = c("default", "MAPE"),
                        ncores = parallel::detectCores(), ...,
                        nfolds = 5, seed, cv.ind, trace = FALSE) {
  #TODO: 
  #   system-specific parallel: Windows parLapply; others: mclapply
  eval.metric <- match.arg(eval.metric)

  max.cores <- parallel::detectCores()
  if (ncores > max.cores) {
    cat("The number of cores specified (", ncores, ") is larger than the number of avaiable cores (", max.cores, "). Use ", max.cores, " cores instead! \n", sep = "")
    ncores = max.cores
  }
  
  fit <- biglasso(X = X, y = y, row.idx = row.idx, ncores = ncores, ...)
  n <- fit$n
  E <- Y <- matrix(NA, nrow=n, ncol=length(fit$lambda))
  # y <- fit$y # this would cause error if eval.metric == "MAPE"
  
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

  parallel <- FALSE
  if (ncores > 1) {
    cluster <- parallel::makeCluster(ncores)
    if (!("cluster" %in% class(cluster))) stop("cluster is not of class 'cluster'; see ?makeCluster")
    parallel <- TRUE
    ## pass the descriptor info to each cluster ##
    xdesc <- bigmemory::describe(X)
    parallel::clusterExport(cluster, c("cv.ind", "xdesc", "y", "cv.args", 
                                       "parallel", "eval.metric"), 
                            envir=environment())
    parallel::clusterCall(cluster, function() {
     
      require(biglasso)
      # require(bigmemory)
      # require(Matrix)
      # dyn.load("~/GitHub/biglasso.Rcheck/biglasso/libs/biglasso.so")
      # source("~/GitHub/biglasso/R/biglasso.R")
      # source("~/GitHub/biglasso/R/predict.R")
      # source("~/GitHub/biglasso/R/loss.R")
    })
    fold.results <- parallel::parLapply(cl = cluster, X = 1:nfolds, fun = cvf, XX = xdesc, 
                                        y = y, eval.metric = eval.metric, 
                                        cv.ind = cv.ind, cv.args = cv.args, 
                                        parallel = parallel)
    parallel::stopCluster(cluster)
  }

  for (i in 1:nfolds) {
    if (parallel) {
      res <- fold.results[[i]]
    } else {
      if (trace) cat("Starting CV fold #", i, sep="", "\n")
      res <- cvf(i, X, y, eval.metric, cv.ind, cv.args)
    }
    E[cv.ind == i, 1:res$nl] <- res$loss
    if (fit$family == "binomial") PE[cv.ind == i, 1:res$nl] <- res$pe
    Y[cv.ind == i, 1:res$nl] <- res$yhat
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

  val <- list(cve=cve, cvse=cvse, lambda=lambda, fit=fit, min=min, lambda.min=lambda[min],
              null.dev=mean(loss.biglasso(y, rep(mean(y), n), 
                                          fit$family, eval.metric = eval.metric)),
              cv.ind = cv.ind,
              eval.metric = eval.metric)
  if (fit$family=="binomial") {
    pe <- apply(PE, 2, mean)
    val$pe <- pe[is.finite(pe)]
  }
  # if (returnY) val$Y <- Y
  structure(val, class=c("cv.biglasso", "cv.ncvreg"))
}

cvf <- function(i, XX, y, eval.metric, cv.ind, cv.args, parallel= FALSE) {
  # reference to the big.matrix by descriptor info
  if (parallel) {
    XX <- attach.big.matrix(XX)
  }
  cv.args$X <- XX
  cv.args$y <- y
  cv.args$row.idx <- which(cv.ind != i)
  cv.args$warn <- FALSE
  cv.args$ncores <- 1

  idx.test <- which(cv.ind == i)
  fit.i <- do.call("biglasso", cv.args)

  y2 <- y[cv.ind==i]
  yhat <- matrix(predict(fit.i, XX, row.idx = idx.test, type="response"), length(y2))

  loss <- loss.biglasso(y2, yhat, fit.i$family, eval.metric = eval.metric)

  pe <- if (fit.i$family=="binomial") {(yhat < 0.5) == y2} else NULL
  list(loss=loss, pe=pe, nl=length(fit.i$lambda), yhat=yhat)
}

