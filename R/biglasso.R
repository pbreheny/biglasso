#' Fit lasso penalized regression path for big data
#' 
#' Extend lasso model fitting to big data that cannot be loaded into memory.
#' Fit solution paths for linear or logistic regression models penalized by
#' lasso, ridge, or elastic-net over a grid of values for the regularization
#' parameter lambda.
#' 
#' The objective function for linear regression (\code{family = "gaussian"}) is
#' \deqn{\frac{1}{2n}\textrm{RSS} + \textrm{penalty},} for logistic regression
#' (\code{family = "binomial"}) it is \deqn{-\frac{1}{n} loglike +
#' \textrm{penalty}.}
#' 
#' Several advanced feature screening rules are implemented. For
#' lasso-penalized linear regression, all the options of \code{screen} are
#' applicable. Our proposal rule - "SSR-BEDPP" - achieves highest speedup so
#' it's the recommended one, especially for ultrahigh-dimensional large-scale
#' data sets. For logistic regression and/or the elastic net penalty, only
#' "SSR" is applicable for now. More efficient rules are under development.
#' 
#' @param X The design matrix, without an intercept. It must be a
#' \code{\link[bigmemory]{big.matrix}} object. The function standardizes the
#' data and includes an intercept internally by default during the model
#' fitting.
#' @param y The response vector.
#' @param row.idx The integer vector of row indices of \code{X} that used for
#' fitting the model. \code{1:nrow(X)} by default.
#' @param penalty The penalty to be applied to the model. Either "lasso" (the
#' default), "ridge", or "enet" (elastic net).
#' @param family Either "gaussian" or "binomial", depending on the response.
#' @param alg.logistic The algorithm used in logistic regression. If "Newton"
#' then the exact hessian is used (default); if "MM" then a
#' majorization-minimization algorithm is used to set an upper-bound on the
#' hessian matrix. This can be faster, particularly in data-larger-than-RAM
#' case.
#' @param screen The feature screening rule used at each \code{lambda} that
#' discards features to speed up computation: "SSR" (default) is the sequential
#' strong rule; "SEDPP" is the (sequential) EDPP rule. "SSR-BEDPP", "SSR-Dome",
#' and "SSR-Slores" are our newly proposed screening rules which combine the
#' strong rule with a safe rule (BEDPP, Dome test, or Slores rule). Among the
#' three, the first two are for lasso-penalized linear regression, and the last
#' one is for lasso-penalized logistic regression. "None" is to not apply a
#' screening rule. \strong{Note that:} (1) for linear regression with elastic
#' net penalty, both "SSR" and "SSR-BEDPP" are applicable since version 1.3-0;
#' (2) only "SSR" is applicable to elastic-net-penalized logistic regression;
#' (3) active set cycling strategy is incorporated with these screening rules
#' by default. All other options with suffix "-NAC" are the corresponding
#' versions without active set cycling update. These rules are for research
#' purpose only.
#' @param safe.thresh the threshold value between 0 and 1 that controls when to
#' stop safe test in the "SSR-Dome" and "SSR-BEDPP" rules. For example, 0.01
#' means to stop Dome test at next lambda iteration if the number of features
#' rejected by safe test at current lambda iteration is not larger than 1\% of
#' the total number of features. So 1 means to always turn off safe test,
#' whereas 0 (default) means to turn off safe test if the number of features
#' rejected by safe test is 0 at current lambda.
#' @param ncores The number of OpenMP threads used for parallel computing.
#' @param alpha The elastic-net mixing parameter that controls the relative
#' contribution from the lasso (l1) and the ridge (l2) penalty. The penalty is
#' defined as \deqn{ \alpha||\beta||_1 + (1-\alpha)/2||\beta||_2^2.}
#' \code{alpha=1} is the lasso penalty, \code{alpha=0} the ridge penalty,
#' \code{alpha} in between 0 and 1 is the elastic-net ("enet") penalty.
#' @param lambda.min The smallest value for lambda, as a fraction of
#' lambda.max.  Default is .001 if the number of observations is larger than
#' the number of covariates and .05 otherwise.
#' @param nlambda The number of lambda values.  Default is 100.
#' @param lambda.log.scale Whether compute the grid values of lambda on log
#' scale (default) or linear scale.
#' @param lambda A user-specified sequence of lambda values.  By default, a
#' sequence of values of length \code{nlambda} is computed, equally spaced on
#' the log scale.
#' @param eps Convergence threshold for inner coordinate descent.  The
#' algorithm iterates until the maximum change in the objective after any
#' coefficient update is less than \code{eps} times the null deviance. Default
#' value is \code{1e-7}.
#' @param max.iter Maximum number of iterations.  Default is 1000.
#' @param dfmax Upper bound for the number of nonzero coefficients.  Default is
#' no upper bound.  However, for large data sets, computational burden may be
#' heavy for models with a large number of nonzero coefficients.
#' @param penalty.factor A multiplicative factor for the penalty applied to
#' each coefficient.  If supplied, \code{penalty.factor} must be a numeric
#' vector of length equal to the number of columns of \code{X}.  The purpose of
#' \code{penalty.factor} is to apply differential penalization if some
#' coefficients are thought to be more likely than others to be in the model.
#' Current package doesn't allow unpenalized coefficients. That
#' is\code{penalty.factor} cannot be 0.
#' @param warn Return warning messages for failures to converge and model
#' saturation?  Default is TRUE.
#' @param output.time Whether to print out the start and end time of the model
#' fitting. Default is FALSE.
#' @param return.time Whether to return the computing time of the model
#' fitting. Default is TRUE.
#' @param verbose Whether to output the timing of each lambda iteration.
#' Default is FALSE.
#' @return An object with S3 class \code{"biglasso"} with following variables.
#' \item{beta}{The fitted matrix of coefficients, store in sparse matrix
#' representation. The number of rows is equal to the number of coefficients,
#' whereas the number of columns is equal to \code{nlambda}.} \item{iter}{A
#' vector of length \code{nlambda} containing the number of iterations until
#' convergence at each value of \code{lambda}.} \item{lambda}{The sequence of
#' regularization parameter values in the path.} \item{penalty}{Same as above.}
#' \item{family}{Same as above.} \item{alpha}{Same as above.} \item{loss}{A
#' vector containing either the residual sum of squares (\code{for "gaussian"})
#' or negative log-likelihood (for \code{"binomial"}) of the fitted model at
#' each value of \code{lambda}.} \item{penalty.factor}{Same as above.}
#' \item{n}{The number of observations used in the model fitting. It's equal to
#' \code{length(row.idx)}.} \item{center}{The sample mean vector of the
#' variables, i.e., column mean of the sub-matrix of \code{X} used for model
#' fitting.} \item{scale}{The sample standard deviation of the variables, i.e.,
#' column standard deviation of the sub-matrix of \code{X} used for model
#' fitting.} \item{y}{The response vector used in the model fitting. Depending
#' on \code{row.idx}, it could be a subset of the raw input of the response
#' vector y.} \item{screen}{Same as above.} \item{col.idx}{The indices of
#' features that have 'scale' value greater than 1e-6. Features with 'scale'
#' less than 1e-6 are removed from model fitting.} \item{rejections}{The number
#' of features rejected at each value of \code{lambda}.}
#' \item{safe_rejections}{The number of features rejected by safe rules at each
#' value of \code{lambda}. Only for "SSR-Dome", "SSR-BEDPP" and "SSR-Slores"
#' cases.}
#' @author Yaohui Zeng and Patrick Breheny
#'
#' Maintainer: Yaohui Zeng <yaohui.zeng@@gmail.com>
#' @seealso \code{\link{biglasso-package}}, \code{\link{setupX}},
#' \code{\link{cv.biglasso}}, \code{\link{plot.biglasso}},
#' \code{\link[ncvreg]{ncvreg}}
#' @examples
#' 
#' ## Linear regression
#' data(colon)
#' X <- colon$X
#' y <- colon$y
#' X.bm <- as.big.matrix(X, backingfile = "")
#' # lasso, default
#' par(mfrow=c(1,2))
#' fit.lasso <- biglasso(X.bm, y, family = 'gaussian')
#' plot(fit.lasso, log.l = TRUE, main = 'lasso')
#' # elastic net
#' fit.enet <- biglasso(X.bm, y, penalty = 'enet', alpha = 0.5, family = 'gaussian')
#' plot(fit.enet, log.l = TRUE, main = 'elastic net, alpha = 0.5')
#' 
#' ## Logistic regression
#' data(colon)
#' X <- colon$X
#' y <- colon$y
#' X.bm <- as.big.matrix(X, backingfile = "")
#' # lasso, default
#' par(mfrow = c(1, 2))
#' fit.bin.lasso <- biglasso(X.bm, y, penalty = 'lasso', family = "binomial")
#' plot(fit.bin.lasso, log.l = TRUE, main = 'lasso')
#' # elastic net
#' fit.bin.enet <- biglasso(X.bm, y, penalty = 'enet', alpha = 0.5, family = "binomial")
#' plot(fit.bin.enet, log.l = TRUE, main = 'elastic net, alpha = 0.5')
#' 
#' @export biglasso
biglasso <- function(X, y, row.idx = 1:nrow(X),
                     penalty = c("lasso", "ridge", "enet"),
                     family = c("gaussian","binomial"), 
                     alg.logistic = c("Newton", "MM"),
                     screen = c("SSR", "SEDPP", "SSR-BEDPP", "SSR-Slores", 
                                "SSR-Dome", "None", "NS-NAC", "SSR-NAC", 
                                "SEDPP-NAC", "SSR-Dome-NAC", "SSR-BEDPP-NAC",
                                "SSR-Slores-NAC"),
                     safe.thresh = 0, ncores = 1, alpha = 1,
                     lambda.min = ifelse(nrow(X) > ncol(X),.001,.05), 
                     nlambda = 100, lambda.log.scale = TRUE,
                     lambda, eps = 1e-7, max.iter = 1000, 
                     dfmax = ncol(X)+1,
                     penalty.factor = rep(1, ncol(X)), 
                     warn = TRUE, output.time = FALSE,
                     return.time = TRUE,
                     verbose = FALSE) {

  family <- match.arg(family)
  penalty <- match.arg(penalty)
  alg.logistic <- match.arg(alg.logistic)
  screen <- match.arg(screen)
  lambda.min <- max(lambda.min, 1.0e-6)
  
  if (identical(penalty, "lasso")) {
    alpha <- 1
  } else if (identical(penalty, 'ridge')) {
    alpha <- 1.0e-6 ## equivalent to ridge regression
  } else if (identical(penalty, 'enet')) {
    if (alpha >= 1 || alpha <= 0) {
      stop("alpha must be between 0 and 1 for elastic net penalty.")
    }
    if (family == 'gaussian' && (!screen %in% c("SSR", "SSR-BEDPP"))) {
      screen <- "SSR"
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
      y <- as.numeric(y == max(y))
    }
    n.pos <- sum(y) # number of 1's
    ylab <- ifelse(y == 0, -1, 1) # response label vector of {-1, 1}
  }

  if (family=="gaussian") {
    yy <- y - mean(y)
  } else {
    yy <- y
  }

  p <- ncol(X)
  if (length(penalty.factor) != p) stop("penalty.factor does not match up with X")
  ## for now penalty.factor is only applicable for "SSR"
  if (screen != "SSR") penalty.factor <- rep(1, p) 
  storage.mode(penalty.factor) <- "double"
  
  n <- length(row.idx) ## subset of X. idx: indices of rows.
  if (missing(lambda)) {
    user.lambda <- FALSE
    lambda <- rep(0.0, nlambda);
  } else {
    nlambda <- length(lambda)
    user.lambda <- TRUE
  }

  ## fit model
  if (output.time) {
    cat("\nStart biglasso: ", format(Sys.time()), '\n')
  }
  if (family == 'gaussian') {
    time <- system.time(
      {
        switch(screen,
               "None" = {
                 res <- .Call("cdfit_gaussian", X@address, yy, as.integer(row.idx-1),
                              lambda, as.integer(nlambda), as.integer(lambda.log.scale),
                              lambda.min, alpha,
                              as.integer(user.lambda | any(penalty.factor==0)),
                              eps, as.integer(max.iter), penalty.factor,
                              as.integer(dfmax), as.integer(ncores), as.integer(verbose),
                              PACKAGE = 'biglasso')
               },
               "SEDPP" = {
                 res <- .Call("cdfit_gaussian_edpp_active", X@address, yy, as.integer(row.idx-1),
                              lambda, as.integer(nlambda), as.integer(lambda.log.scale),
                              lambda.min, alpha,
                              as.integer(user.lambda | any(penalty.factor==0)),
                              eps, as.integer(max.iter), penalty.factor,
                              as.integer(dfmax), as.integer(ncores),
                              PACKAGE = 'biglasso')
               },
               "SSR" = {
                 res <- .Call("cdfit_gaussian_hsr", X@address, yy, as.integer(row.idx-1),
                              lambda, as.integer(nlambda), as.integer(lambda.log.scale),
                              lambda.min, alpha,
                              as.integer(user.lambda | any(penalty.factor==0)),
                              eps, as.integer(max.iter), penalty.factor,
                              as.integer(dfmax), as.integer(ncores), as.integer(verbose),
                              PACKAGE = 'biglasso')
               },
               "SSR-Dome" = {
                 res <- .Call("cdfit_gaussian_hsr_dome", X@address, yy, as.integer(row.idx-1),
                              lambda, as.integer(nlambda), as.integer(lambda.log.scale),
                              lambda.min, alpha,
                              as.integer(user.lambda | any(penalty.factor==0)),
                              eps, as.integer(max.iter), penalty.factor,
                              as.integer(dfmax), as.integer(ncores), safe.thresh, 
                              as.integer(verbose),
                              PACKAGE = 'biglasso')
               },
               "SSR-BEDPP" = {
                 res <- .Call("cdfit_gaussian_hsr_bedpp", X@address, yy, as.integer(row.idx-1),
                              lambda, as.integer(nlambda), as.integer(lambda.log.scale),
                              lambda.min, alpha,
                              as.integer(user.lambda | any(penalty.factor==0)),
                              eps, as.integer(max.iter), penalty.factor,
                              as.integer(dfmax), as.integer(ncores), safe.thresh, 
                              as.integer(verbose),
                              PACKAGE = 'biglasso')
               },
               "NS-NAC" = {
                 res <- .Call("cdfit_gaussian_nac", X@address, yy, as.integer(row.idx-1),
                              lambda, as.integer(nlambda), as.integer(lambda.log.scale),
                              lambda.min, alpha,
                              as.integer(user.lambda | any(penalty.factor==0)),
                              eps, as.integer(max.iter), penalty.factor,
                              as.integer(dfmax), as.integer(ncores), as.integer(verbose),
                              PACKAGE = 'biglasso')
               },
               "SEDPP-NAC" = {
                 res <- .Call("cdfit_gaussian_edpp", X@address, yy, as.integer(row.idx-1),
                              lambda, as.integer(nlambda), as.integer(lambda.log.scale),
                              lambda.min, alpha,
                              as.integer(user.lambda | any(penalty.factor==0)),
                              eps, as.integer(max.iter), penalty.factor,
                              as.integer(dfmax), as.integer(ncores),
                              PACKAGE = 'biglasso')
               },
               "SSR-NAC" = {
                 res <- .Call("cdfit_gaussian_hsr_nac", X@address, yy, as.integer(row.idx-1),
                              lambda, as.integer(nlambda), as.integer(lambda.log.scale),
                              lambda.min, alpha,
                              as.integer(user.lambda | any(penalty.factor==0)),
                              eps, as.integer(max.iter), penalty.factor,
                              as.integer(dfmax), as.integer(ncores), as.integer(verbose),
                              PACKAGE = 'biglasso')
               },
               "SSR-Dome-NAC" = {
                 res <- .Call("cdfit_gaussian_hsr_dome_nac", X@address, yy, as.integer(row.idx-1),
                              lambda, as.integer(nlambda), as.integer(lambda.log.scale),
                              lambda.min, alpha,
                              as.integer(user.lambda | any(penalty.factor==0)),
                              eps, as.integer(max.iter), penalty.factor,
                              as.integer(dfmax), as.integer(ncores), safe.thresh, 
                              as.integer(verbose),
                              PACKAGE = 'biglasso')
               },
               "SSR-BEDPP-NAC" = {
                 res <- .Call("cdfit_gaussian_hsr_bedpp_nac", X@address, yy, as.integer(row.idx-1),
                              lambda, as.integer(nlambda), as.integer(lambda.log.scale),
                              lambda.min, alpha,
                              as.integer(user.lambda | any(penalty.factor==0)),
                              eps, as.integer(max.iter), penalty.factor,
                              as.integer(dfmax), as.integer(ncores), safe.thresh, 
                              as.integer(verbose),
                              PACKAGE = 'biglasso')
               },
               stop("Invalid screening method!")
               )
      }
    )
    
    a <- rep(mean(y), nlambda)
    b <- Matrix(res[[1]], sparse = T)
    center <- res[[2]]
    scale <- res[[3]]
    lambda <- res[[4]]
    loss <- res[[5]]
    iter <- res[[6]]
    rejections <- res[[7]]
    
    if (screen %in% c("SSR-Dome", "SSR-BEDPP", "SSR-Dome-NAC", "SSR-BEDPP-NAC")) {
      safe_rejections <- res[[8]]
      col.idx <- res[[9]]
    } else {
      col.idx <- res[[8]]
    }
   
  } else if (family == 'binomial') {
    
    time <- system.time(
      if (alg.logistic == 'MM') {
        res <- .Call("cdfit_binomial_hsr_approx", X@address, yy, as.integer(row.idx-1), 
                     lambda, as.integer(nlambda), lambda.min, alpha, 
                     as.integer(user.lambda | any(penalty.factor==0)),
                     eps, as.integer(max.iter), penalty.factor, 
                     as.integer(dfmax), as.integer(ncores), as.integer(warn),
                     as.integer(verbose),
                     PACKAGE = 'biglasso')
      } else {
        if (screen == "SSR-Slores") {
          res <- .Call("cdfit_binomial_hsr_slores", X@address, yy, as.integer(n.pos),
                       as.integer(ylab), as.integer(row.idx-1), 
                       lambda, as.integer(nlambda), as.integer(lambda.log.scale),
                       lambda.min, alpha, as.integer(user.lambda | any(penalty.factor==0)),
                       eps, as.integer(max.iter), penalty.factor, 
                       as.integer(dfmax), as.integer(ncores), as.integer(warn), safe.thresh,
                       as.integer(verbose),
                       PACKAGE = 'biglasso')
        } else if (screen == "SSR-Slores-NAC") {
          res <- .Call("cdfit_binomial_hsr_slores_nac", X@address, yy, as.integer(n.pos),
                       as.integer(ylab), as.integer(row.idx-1), 
                       lambda, as.integer(nlambda), as.integer(lambda.log.scale),
                       lambda.min, alpha, as.integer(user.lambda | any(penalty.factor==0)),
                       eps, as.integer(max.iter), penalty.factor, 
                       as.integer(dfmax), as.integer(ncores), as.integer(warn), safe.thresh,
                       as.integer(verbose),
                       PACKAGE = 'biglasso')
        } 
        else {
          res <- .Call("cdfit_binomial_hsr", X@address, yy, as.integer(row.idx-1), 
                       lambda, as.integer(nlambda), as.integer(lambda.log.scale),
                       lambda.min, alpha, as.integer(user.lambda | any(penalty.factor==0)),
                       eps, as.integer(max.iter), penalty.factor, 
                       as.integer(dfmax), as.integer(ncores), as.integer(warn),
                       as.integer(verbose),
                       PACKAGE = 'biglasso')
        }
      }
    )
   
    a <- res[[1]]
    b <- Matrix(res[[2]], sparse = T)
    center <- res[[3]]
    scale <- res[[4]]
    lambda <- res[[5]]
    loss <- res[[6]]
    iter <- res[[7]]
    rejections <- res[[8]]
    
    if (screen %in% c("SSR-Slores", "SSR-Slores-NAC")) {
      safe_rejections <- res[[9]]
      col.idx <- res[[10]]
    } else {
      col.idx <- res[[9]]
    }
    
  } else {
    stop("Current version only supports Gaussian or Binominal response!")
  }
  if (output.time) {
    cat("\nEnd biglasso: ", format(Sys.time()), '\n')
  }
  # p.keep <- length(col.idx)
  col.idx <- col.idx + 1 # indices (in R) for which variables have scale > 1e-6

  ## Eliminate saturated lambda values, if any
  ind <- !is.na(iter)
  a <- a[ind]
  b <- b[, ind, drop=FALSE]
  iter <- iter[ind]
  lambda <- lambda[ind]
  loss <- loss[ind]

  if (warn & any(iter==max.iter)) warning("Algorithm failed to converge for some values of lambda")

  ## Unstandardize coefficients:
  beta <- Matrix(0, nrow = (p+1), ncol = length(lambda), sparse = T)
  bb <- b / scale[col.idx]
  beta[col.idx+1, ] <- bb
  beta[1,] <- a - crossprod(center[col.idx], bb)

  ## Names
  varnames <- if (is.null(colnames(X))) paste("V", 1:p, sep="") else colnames(X)
  varnames <- c("(Intercept)", varnames)
  dimnames(beta) <- list(varnames, round(lambda, digits = 4))

  ## Output
  return.val <- list(
    beta = beta,
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
    y = yy,
    screen = screen,
    col.idx = col.idx,
    rejections = rejections
  )
  
  if (screen %in% c("SSR-Dome", "SSR-Dome-NAC", 
                    "SSR-BEDPP", "SSR-BEDPP-NAC", 
                    "SSR-Slores", "SSR-Slores-NAC")) {
    return.val$safe_rejections <- safe_rejections
  } 
  if (return.time) return.val$time <- as.numeric(time['elapsed'])
  
  val <- structure(return.val, class = c("biglasso", 'ncvreg'))
  val
}
