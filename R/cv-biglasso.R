#' Cross-validation for biglasso
#'
#' Perform k-fold cross validation for penalized regression models over a grid
#' of values for the regularization parameter lambda.
#'
#' The function calls `biglasso` `nfolds` times, each time leaving
#' out 1/`nfolds` of the data. The cross-validation error is based on the
#' residual sum of squares when `family="gaussian"` and the binomial
#' deviance when `family="binomial"`.
#'
#' The S3 class object `cv.biglasso` inherits class [ncvreg::cv.ncvreg()]. So S3
#' functions such as `"summary", "plot"` can be directly applied to the
#' `cv.biglasso` object.
#'
#' @param X The design matrix, without an intercept, as in [biglasso()].
#' @param y The response vector, as in `biglasso`.
#' @param row.idx The integer vector of row indices of `X` that used for
#'   fitting the model. as in `biglasso`.
#' @param family Either `"gaussian"`, `"binomial"`, `"cox"` or
#'   `"mgaussian"` depending on the response. `"cox"` and `"mgaussian"`
#'   are not supported yet.
#' @param eval.metric The evaluation metric for the cross-validated error and
#'   for choosing optimal `lambda`. "default" for linear regression is MSE
#'   (mean squared error), for logistic regression is binomial deviance.
#'   "MAPE", for linear regression only, is the Mean Absolute Percentage Error.
#'   "auc", for binary classification, is the area under the receiver operating
#'   characteristic curve (ROC).
#'   "class", for binary classification, gives the misclassification error.
#' @param ncores The number of cores to use for parallel execution of the
#'   cross-validation folds, run on a cluster created by the `parallel`
#'   package. (This is also supplied to the `ncores` argument in
#'   [biglasso()], which is the number of OpenMP threads, but only for
#'   the first call of [biglasso()] that is  run on the entire data. The
#'   individual calls of [biglasso()] for the CV folds are run without
#'   the `ncores` argument.)
#' @param ... Additional arguments to `biglasso`.
#' @param nfolds The number of cross-validation folds.  Default is 5.
#' @param seed The seed of the random number generator in order to obtain
#'   reproducible results.
#' @param cv.ind Which fold each observation belongs to.  By default the
#'   observations are randomly assigned by `cv.biglasso`.
#' @param trace If set to TRUE, cv.biglasso will inform the user of its
#'   progress by announcing the beginning of each CV fold.  Default is FALSE.
#' @param grouped Whether to calculate CV standard error (`cvse`) over
#'   CV folds (`TRUE`), or over all cross-validated predictions. Ignored
#'   when `eval.metric` is 'auc'.
#' 
#' @returns An object with S3 class `"cv.biglasso"` which inherits from
#' class `"cv.ncvreg"`. The following variables are contained in the
#' class (adopted from [ncvreg::cv.ncvreg()]).
#' \item{cve}{The error for each value of `lambda`, averaged across the cross-validation folds.}
#' \item{cvse}{The estimated standard error associated with each value of for `cve`.}
#' \item{lambda}{The sequence of regularization parameter values along which the cross-validation error was calculated.}
#' \item{fit}{The fitted `biglasso` object for the whole data.}
#' \item{min}{The index of `lambda` corresponding to `lambda.min`.}
#' \item{lambda.min}{The value of `lambda` with the minimum cross-validation error.}
#' \item{lambda.1se}{The largest value of `lambda` for which the cross-validation error is at most one standard error larger than the minimum cross-validation error.}
#' \item{null.dev}{The deviance for the intercept-only model.}
#' \item{pe}{If `family="binomial"`, the cross-validation prediction error for each value of `lambda`.}
#' \item{cv.ind}{Same as above.}
#' 
#' @author Yaohui Zeng and Patrick Breheny
#'
#' @seealso [biglasso()], [plot.cv.biglasso()], [summary.cv.biglasso()], [setupX()]
#' 
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
#' @export

cv.biglasso <- function(X, y, row.idx = 1:nrow(X),
                        family = c("gaussian", "binomial", "cox", "mgaussian"),
                        eval.metric = c("default", "MAPE", "auc", "class"),
                        ncores = parallel::detectCores(), ...,
                        nfolds = 5, seed, cv.ind, trace = FALSE, grouped = TRUE) {

  family <- match.arg(family)
  if(!family %in% c("gaussian", "binomial")) stop("CV method for this family not supported yet.")

  #TODO:
  #   system-specific parallel: Windows parLapply; others: mclapply
  eval.metric <- match.arg(eval.metric)

  max.cores <- parallel::detectCores()
  if (ncores > max.cores) {
    cat("The number of cores specified (", ncores, ") is larger than the number of avaiable cores (", max.cores, "). Use ", max.cores, " cores instead! \n", sep = "")
    ncores = max.cores
  }

  fit <- biglasso(X = X, y = y, row.idx = row.idx, family = family, ncores = ncores, ...)

  
  if (eval.metric == "auc") grouped <- TRUE

  E <- matrix(Inf, nrow=if (grouped) nfolds else fit$n, ncol=length(fit$lambda))
  # y <- fit$y # this would cause error if eval.metric == "MAPE"

  if (fit$family == 'binomial') {
    PE <- matrix(NA, nrow=fit$n, ncol=length(fit$lambda))
  }

  if (!missing(seed)) {
    original_seed <- .GlobalEnv$.Random.seed
    on.exit(.GlobalEnv$.Random.seed <- original_seed)
    set.seed(seed)
  }

  n <- fit$n
  if (missing(cv.ind)) {
    if (fit$family=="binomial" && (min(table(y)) > nfolds)) {
      ind1 <- which(y==1)
      ind0 <- which(y==0)
      n1 <- length(ind1)
      n0 <- length(ind0)
      cv.ind1 <- ceiling(sample(n1)/n1*nfolds)
      cv.ind0 <- ceiling(sample(n0)/n0*nfolds)
      cv.ind <- numeric(n)
      cv.ind[y==1] <- cv.ind1
      cv.ind[y==0] <- cv.ind0
    } else {
      cv.ind <- ceiling(sample(n)/n*nfolds)
    }
  }

  cv.args <- list(...)
  cv.args$lambda <- fit$lambda
  cv.args$family <- family

  parallel <- FALSE
  if (ncores > 1) {
    cluster <- parallel::makeCluster(ncores)
    if (!("cluster" %in% class(cluster))) stop("cluster is not of class 'cluster'; see ?makeCluster")
    parallel <- TRUE
    ## pass the descriptor info to each cluster ##
    xdesc <- bigmemory::describe(X)
    parallel::clusterExport(cluster, c("cv.ind", "xdesc", "y", "cv.args",
                                       "parallel", "eval.metric", "biglasso_fit"),
                            envir=environment())
    parallel::clusterCall(cluster, function() {
      require(biglasso)
    })
    fold.results <- parallel::parLapply(cl = cluster,
                                        X = seq_len(nfolds),
                                        fun = cvf,
                                        XX = xdesc,
                                        y = y,
                                        eval.metric = eval.metric,
                                        cv.ind = cv.ind, 
                                        cv.args = cv.args,
                                        grouped = grouped,
                                        parallel = parallel)
    parallel::stopCluster(cluster)
  }

  result.ind <- if (grouped) seq_len(nfolds) else cv.ind
  for (i in seq_len(nfolds)) {
    if (parallel) {
      res <- fold.results[[i]]
    } else {
      if (trace) cat("Starting CV fold #", i, sep="", "\n")
      res <- cvf(i, X, y, eval.metric, cv.ind, cv.args, grouped = grouped)
    }
    E[result.ind == i, match(res$ls, fit$lambda)] <- res$loss
    if (fit$family == "binomial") PE[cv.ind == i, match(res$ls, fit$lambda)] <- res$pe
  }

  ## Eliminate saturated lambda values, if any
  # deliberately do *not* eliminate NAs, which can occur when AUC breaks
  ind <- which(apply(!is.infinite(E), 2, all))
  E <- E[,ind]
  lambda <- fit$lambda[ind]

  ## Return
  if (grouped) {
    foldweights <- sapply(seq_len(nfolds), function(i) sum(cv.ind == i))
    cve <- apply(E, 2, weighted.mean, w = foldweights, na.rm = TRUE)  # na.rm only relevant for 'auc' case
    cvse <- sqrt(apply(sweep(E, 2, cve)^2, 2, weighted.mean, w = foldweights, na.rm = TRUE) / (nfolds - 1))
  } else {
    cve <- apply(E, 2, mean)
    cvse <- apply(E, 2, sd) / sqrt(n)
  }
  # get the index for which cvm is minimal (or maximal, for AUC).
  cve.minimize <- if (eval.metric == "auc") -cve else cve
  # break ties with the smaller index, since we prefer larger values of lambda, all else being equal.
  min <- which.min(cve.minimize)
  lambda.1se <- max(lambda[cve.minimize <= cve.minimize[[min]] + cvse[[min]]], na.rm = TRUE)

  val <- list(cve=cve, cvse=cvse, lambda=lambda, fit=fit, min=min, lambda.min=lambda[min], lambda.1se = lambda.1se,
              null.dev=mean(loss.biglasso(y, rep(mean(y), n),
                                          fit$family, eval.metric = eval.metric)),
              cv.ind = cv.ind,
              eval.metric = eval.metric)
  if (fit$family=="binomial") {
    pe <- apply(PE, 2, mean)
    val$pe <- pe[is.finite(pe)]
  }
  structure(val, class=c("cv.biglasso", "cv.ncvreg"))
}

cvf <- function(i, XX, y, eval.metric, cv.ind, cv.args, grouped, parallel= FALSE) {
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

  loss <- loss.biglasso(y2, yhat, fit.i$family, eval.metric = eval.metric, grouped = grouped)

  pe <- if (fit.i$family=="binomial") (yhat <= 0.5) == y2

  list(loss=loss, pe=pe, ls=fit.i$lambda, yhat=yhat)
}

