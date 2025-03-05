#' Fit lasso penalized regression path for big data
#' 
#' Extend lasso model fitting to big data that cannot be loaded into memory.
#' Fit solution paths for linear, logistic or Cox regression models penalized by
#' lasso, ridge, or elastic-net over a grid of values for the regularization
#' parameter lambda.
#' 
#' The objective function for linear regression or multiple responses linear regression 
#' (`family = "gaussian"` or `family = "mgaussian"`) is
#' \deqn{\frac{1}{2n}\textrm{RSS} + \lambda*\textrm{penalty},}{(1/(2n))*RSS+
#' \lambda*penalty,}
#' where for `family = "mgaussian"`), a group-lasso type penalty is applied.
#' For logistic regression
#' (`family = "binomial"`) it is \deqn{-\frac{1}{n} loglike +
#' \lambda*\textrm{penalty},}{-(1/n)*loglike+\lambda*penalty}, for cox regression,
#'  breslow approximation for ties is applied.
#' 
#' Several advanced feature screening rules are implemented. For
#' lasso-penalized linear regression, all the options of `screen` are
#' applicable. Our proposal adaptive rule - `"Adaptive"` - achieves highest speedup
#' so it's the recommended one, especially for ultrahigh-dimensional large-scale
#' data sets. For cox regression and/or the elastic net penalty, only
#' `"SSR"` is applicable for now. More efficient rules are under development.
#' 
#' @param X The design matrix, without an intercept. It must be a
#' double type [bigmemory::big.matrix()] object. The function
#' standardizes the data and includes an intercept internally by default during
#' the model fitting.
#' @param y The response vector for `family="gaussian"` or `family="binomial"`.
#' For `family="cox"`, `y` should be a two-column matrix with columns
#' 'time' and 'status'. The latter is a binary variable, with '1' indicating death,
#'  and '0' indicating right censored. For `family="mgaussin"`, `y`
#'  should be a n*m matrix where n is the sample size and m is the number of
#'  responses.
#' @param row.idx The integer vector of row indices of `X` that used for
#' fitting the model. `1:nrow(X)` by default.
#' @param penalty The penalty to be applied to the model. Either `"lasso"`
#' (the default), `"ridge"`, or `"enet"` (elastic net).
#' @param family Either `"gaussian"`, `"binomial"`, `"cox"` or
#' `"mgaussian"` depending on the response.
#' @param alg.logistic The algorithm used in logistic regression. If "Newton"
#' then the exact hessian is used (default); if "MM" then a
#' majorization-minimization algorithm is used to set an upper-bound on the
#' hessian matrix. This can be faster, particularly in data-larger-than-RAM
#' case.
#' @param screen The feature screening rule used at each `lambda` that
#' discards features to speed up computation: `"SSR"` (default if
#' `penalty="ridge"` or `penalty="enet"` )is the sequential strong rule;
#' `"Hybrid"` is our newly proposed hybrid screening rules which combine the
#' strong rule with a safe rule. `"Adaptive"` (default for `penalty="lasso"`
#' without `penalty.factor`) is our newly proposed adaptive rules which
#' reuse screening reference for multiple lambda values. \strong{Note that:}
#' (1) for linear regression with elastic net penalty, both `"SSR"` and
#' `"Hybrid"` are applicable since version 1.3-0;  (2) only `"SSR"` is
#' applicable to elastic-net-penalized logistic regression or cox regression;
#' (3) active set cycling strategy is incorporated with these screening rules.
#' @param safe.thresh the threshold value between 0 and 1 that controls when to
#' stop safe test. For example, 0.01 means to stop safe test at next lambda 
#' iteration if the number of features rejected by safe test at current lambda
#' iteration is not larger than 1\% of the total number of features. So 1 means
#' to always turn off safe test, whereas 0 (default) means to turn off safe test
#' if the number of features rejected by safe test is 0 at current lambda.
#' @param update.thresh the non negative threshold value that controls how often to
#' update the reference of safe rules for "Adaptive" methods. Smaller value means
#' updating more often.
#' @param ncores The number of OpenMP threads used for parallel computing.
#' @param alpha The elastic-net mixing parameter that controls the relative
#' contribution from the lasso (l1) and the ridge (l2) penalty. The penalty is
#' defined as \deqn{ \alpha||\beta||_1 + (1-\alpha)/2||\beta||_2^2.}
#' `alpha=1` is the lasso penalty, `alpha=0` the ridge penalty,
#' `alpha` in between 0 and 1 is the elastic-net ("enet") penalty.
#' @param lambda.min The smallest value for lambda, as a fraction of
#' lambda.max.  Default is .001 if the number of observations is larger than
#' the number of covariates and .05 otherwise.
#' @param nlambda The number of lambda values.  Default is 100.
#' @param lambda.log.scale Whether compute the grid values of lambda on log
#' scale (default) or linear scale.
#' @param lambda A user-specified sequence of lambda values.  By default, a
#' sequence of values of length `nlambda` is computed, equally spaced on
#' the log scale.
#' @param eps Convergence threshold for inner coordinate descent.  The
#' algorithm iterates until the maximum change in the objective after any
#' coefficient update is less than `eps` times the null deviance. Default
#' value is `1e-7`.
#' @param max.iter Maximum number of iterations.  Default is 1000.
#' @param dfmax Upper bound for the number of nonzero coefficients.  Default is
#' no upper bound.  However, for large data sets, computational burden may be
#' heavy for models with a large number of nonzero coefficients.
#' @param penalty.factor A multiplicative factor for the penalty applied to
#' each coefficient. If supplied, `penalty.factor` must be a numeric
#' vector of length equal to the number of columns of `X`.  The purpose of
#' `penalty.factor` is to apply differential penalization if some
#' coefficients are thought to be more likely than others to be in the model.
#' Current package doesn't allow unpenalized coefficients. That
#' is`penalty.factor` cannot be 0. `penalty.factor` is only supported
#' for "SSR" screen.
#' @param warn Return warning messages for failures to converge and model
#' saturation?  Default is TRUE.
#' @param output.time Whether to print out the start and end time of the model
#' fitting. Default is FALSE.
#' @param return.time Whether to return the computing time of the model
#' fitting. Default is TRUE.
#' @param verbose Whether to output the timing of each lambda iteration.
#' Default is FALSE.
#' 
#' @returns An object with S3 class `"biglasso"` for
#' `"gaussian", "binomial", "cox"` families, or an object with S3 class
#' `"mbiglasso"` for `"mgaussian"` family,  with following variables.
#' \item{beta}{The fitted matrix of coefficients, store in sparse matrix
#' representation. The number of rows is equal to the number of coefficients,
#' whereas the number of columns is equal to `nlambda`. For `"mgaussian"`
#' family with m responses, it is a list of m such matrices.} 
#' \item{iter}{A vector of length `nlambda` containing the number of 
#' iterations until convergence at each value of `lambda`.} 
#' \item{lambda}{The sequence of regularization parameter values in the path.}
#' \item{penalty}{Same as above.}
#' \item{family}{Same as above.}
#' \item{alpha}{Same as above.} 
#' \item{loss}{A vector containing either the residual sum of squares 
#' (for `"gaussian", "mgaussian"`) or negative log-likelihood
#' (for `"binomial", "cox"`) of the fitted model at each value of `lambda`.}
#' \item{penalty.factor}{Same as above.}
#' \item{n}{The number of observations used in the model fitting. It's equal to
#' `length(row.idx)`.} 
#' \item{center}{The sample mean vector of the variables, i.e., column mean of
#' the sub-matrix of `X` used for model fitting.} 
#' \item{scale}{The sample standard deviation of the variables, i.e., column
#' standard deviation of the sub-matrix of `X` used for model fitting.} 
#' \item{y}{The response vector used in the model fitting. Depending on
#' `row.idx`, it could be a subset of the raw input of the response vector y.}
#' \item{screen}{Same as above.} 
#' \item{col.idx}{The indices of features that have 'scale' value greater than
#' 1e-6. Features with 'scale' less than 1e-6 are removed from model fitting.} 
#' \item{rejections}{The number of features rejected at each value of `lambda`.}
#' \item{safe_rejections}{The number of features rejected by safe rules at each
#' value of `lambda`.}
#' 
#' @author Yaohui Zeng, Chuyi Wang and Patrick Breheny
#'
#' @references
#' Zeng Y and Breheny P. (2021) The biglasso Package: A Memory- and Computation-
#' Efficient Solver for Lasso Model Fitting with Big Data in R.
#' *R Journal*, **12**: 6-19.
#' \doi{10.32614/RJ-2021-001}
#' 
#' @seealso [biglasso-package], [setupX()], [cv.biglasso()], [plot.biglasso()], [ncvreg::ncvreg()]
#' 
#' @examples
#' ## Linear regression
#' data(colon)
#' X <- colon$X
#' y <- colon$y
#' X.bm <- as.big.matrix(X)
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
#' X.bm <- as.big.matrix(X)
#' # lasso, default
#' par(mfrow = c(1, 2))
#' fit.bin.lasso <- biglasso(X.bm, y, penalty = 'lasso', family = "binomial")
#' plot(fit.bin.lasso, log.l = TRUE, main = 'lasso')
#' # elastic net
#' fit.bin.enet <- biglasso(X.bm, y, penalty = 'enet', alpha = 0.5, family = "binomial")
#' plot(fit.bin.enet, log.l = TRUE, main = 'elastic net, alpha = 0.5')
#' 
#' ## Cox regression
#' set.seed(10101)
#' N <- 1000; p <- 30; nzc <- p/3
#' X <- matrix(rnorm(N * p), N, p)
#' beta <- rnorm(nzc)
#' fx <- X[, seq(nzc)] %*% beta/3
#' hx <- exp(fx)
#' ty <- rexp(N, hx)
#' tcens <- rbinom(n = N, prob = 0.3, size = 1)  # censoring indicator
#' y <- cbind(time = ty, status = 1 - tcens)  # y <- Surv(ty, 1 - tcens) with library(survival)
#' X.bm <- as.big.matrix(X)
#' fit <- biglasso(X.bm, y, family = "cox")
#' plot(fit, main = "cox")
#' 
#' ## Multiple responses linear regression
#' set.seed(10101)
#' n=300; p=300; m=5; s=10; b=1
#' x = matrix(rnorm(n * p), n, p)
#' beta = matrix(seq(from=-b,to=b,length.out=s*m),s,m)
#' y = x[,1:s] %*% beta + matrix(rnorm(n*m,0,1),n,m)
#' x.bm = as.big.matrix(x)
#' fit = biglasso(x.bm, y, family = "mgaussian")
#' plot(fit, main = "mgaussian")
#' 
#' @export biglasso
biglasso <- function(X, y, row.idx = 1:nrow(X),
                     penalty = c("lasso", "ridge", "enet"),
                     family = c("gaussian", "binomial", "cox", "mgaussian"), 
                     alg.logistic = c("Newton", "MM"),
                     screen = c("Adaptive", "SSR", "Hybrid", "None"),
                     safe.thresh = 0, update.thresh = 1, ncores = 1, alpha = 1,
                     lambda.min = ifelse(nrow(X) > ncol(X),.001,.05), 
                     nlambda = 100, lambda.log.scale = TRUE,
                     lambda, eps = 1e-7, max.iter = 1000, 
                     dfmax = ncol(X)+1,
                     penalty.factor = rep(1, ncol(X)),
                     warn = TRUE, output.time = FALSE,
                     return.time = TRUE,
                     verbose = FALSE) {
  
  # set up defaults -------------------------------------------------------
  
  # Match deprecated screen methods 
  if(length(screen) == 1 &&
     screen %in% c("SEDPP", "SSR-BEDPP", "SSR-Slores", "SSR-Slores-Batch", 
                   "SSR-Dome", "None", "NS-NAC", "SSR-NAC", "SEDPP-NAC",
                   "SSR-Dome-NAC", "SSR-BEDPP-NAC", "SSR-Slores-NAC", 
                   "SEDPP-Batch", "SEDPP-Batch-SSR", "SEDPP-Batchfix-SSR")) {
  
    if(screen %in% c("SSR-BEDPP", "SSR-Slores")) {
      warning("Hybrid screen methods (\"SSR-BEDPP\", \"SSR-Slores\") will be renamed as \"Hybrid\". Automatically switching to \"Hybrid\" screen method")
      screen = "Hybrid"
    }
    if(screen %in% c("SSR-Slores-Batch", "SEDPP-Batch", "SEDPP-Batch-SSR", "SEDPP-Batchfix-SSR")) {
      warning("Adaptive or batch screen methods (\"SSR-Slores-Batch\", \"SEDPP-Batch\", \"SEDPP-Batch-SSR\", \"SEDPP-Batchfix-SSR\") will be renamed as \"Adaptive\". Automatically switching to \"Adaptive\" screen method.")
      screen = "Adaptive"
    }
    if(screen %in% c("SEDPP", "SSR-Dome", "NS-NAC", "SSR-NAC", "SEDPP-NAC",
                     "SSR-Dome-NAC", "SSR-BEDPP-NAC", "SSR-Slores-NAC")){
      warning("The following screen methods will be removed:\n\"SEDPP\", \"SSR-Dome\", \"NS-NAC\", \"SSR-NAC\", \"SEDPP-NAC\",\"SSR-Dome-NAC\", \"SSR-BEDPP-NAC\", \"SSR-Slores-NAC\".\nAutomatically switching to \"Adaptive\" screen method.")
      screen = "Adaptive"
    }
  }
  

  family <- match.arg(family)
  penalty <- match.arg(penalty)
  alg.logistic <- match.arg(alg.logistic)
  if (!identical(penalty, "lasso") || any(penalty.factor != 1) || alg.logistic =="MM"){
    if(length(screen) == 1) screen <- match.arg(screen, choices = c("SSR", "Adaptive", "Hybrid", "None"))
    else screen <- "SSR"
  } else if (family == "cox") {
    if(length(screen) == 1) screen <- match.arg(screen, choices = c("SSR", "Adaptive", "Hybrid", "None", "scox", "sscox", "safe"))
    else screen <- "SSR"
  } else {
    screen <- match.arg(screen)
  }
  lambda.min <- max(lambda.min, 1.0e-6)
  

  if (identical(penalty, "lasso")) {
    alpha <- 1
  } else if (identical(penalty, 'ridge')) {
    alpha <- 1.0e-6 ## equivalent to ridge regression
    if (screen == "Adaptive") {
      warning("For now \"ridge\" does not support \"Adaptive\" screen. Automatically switching to \"SSR\"." )
      screen <- "SSR"
    }
  } else if (identical(penalty, 'enet')) {
    if (alpha >= 1 || alpha <= 0) {
      stop("alpha must be between 0 and 1 for elastic net penalty.")
    }
    if (screen == "Adaptive") {
      warning("For now \"enet\" does not support \"Adaptive\" screen. Automatically switching to \"SSR\"." )
      screen <- "SSR"
    } 
  }

  if (!("big.matrix" %in% class(X)) || typeof(X) != "double") stop("X must be a double type big.matrix.")
  if (nlambda < 2) stop("nlambda must be at least 2")
  # subset of the response vector
  if (is.matrix(y)) y <- y[row.idx,]
  else y <- y[row.idx]

  if (any(is.na(y))) stop("Missing data (NA's) detected.  Take actions (e.g., removing cases, removing features, imputation) to eliminate missing data before fitting the model.")

  if (!is.double(y)) {
    if (is.matrix(y)) tmp <- try(storage.mode(y) <- "numeric", silent=TRUE)
    else tmp <- try(y <- as.numeric(y), silent=TRUE)
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
  
  if (family == 'cox') {
    if (!is.matrix(y)) stop("y must be a matrix or able to be coerced to a matrix")
    if (ncol(y) != 2) stop("y must have two columns for survival data: time-on-study and a censoring indicator")
    if (!all(y[,2] %in% c(0,1))) stop("Second column of y must be a binary censoring indicator")
    if (!any(y[,2] > 0)) stop('Require at least one failure')
    tOrder = order(y[,1])
    d <- as.numeric(table(y[y[,2]==1,1]))
    dtime <- sort(unique(y[y[,2]==1,1]))
    row.idx.cox <- which(y[tOrder,1] >= min (dtime))
    d_idx <- integer(length(row.idx.cox))
    for(i in 1:length(row.idx.cox)) d_idx[i] <- max(which(dtime <= y[tOrder[row.idx.cox[i]],1])) 
  }

  if (family == "gaussian") {
    yy <- y - mean(y)
  } else if (family == "binomial") {
    yy <- y
  } else if (family == "cox") {
    yy <- y[tOrder[row.idx.cox],2]
  } else if (family == "mgaussian") {
    yy <- t(scale(y, scale = F))
  }

  p <- ncol(X)
  if (length(penalty.factor) != p) stop("penalty.factor does not match up with X")
  ## for now penalty.factor is only applicable for "SSR"
  if(any(penalty.factor != 1) & screen != "SSR") {
    warning("For now penalty.factor is only applicable for \"SSR\". Automatically switching to \"SSR\".")
    screen = "SSR"
  }
  storage.mode(penalty.factor) <- "double"
  
  n <- length(row.idx) ## subset of X. idx: indices of rows.
  if (missing(lambda)) {
    user.lambda <- FALSE
    lambda <- rep(0.0, nlambda);
  } else {
    nlambda <- length(lambda)
    user.lambda <- TRUE
  }

  ## fit model ----------------------------------------------------------
  if (output.time) {
    cat("\nStart biglasso: ", format(Sys.time()), '\n')
  }
  if (family == 'gaussian') {
    time <- system.time(
      {
        switch(screen,
               "Adaptive" = {
                 res <- .Call("cdfit_gaussian_ada_edpp_ssr", X@address, yy, as.integer(row.idx-1),
                              lambda, as.integer(nlambda), as.integer(lambda.log.scale),
                              lambda.min, alpha,
                              as.integer(user.lambda | any(penalty.factor==0)),
                              eps, as.integer(max.iter), penalty.factor,
                              as.integer(dfmax), as.integer(ncores), update.thresh, as.integer(verbose),
                              PACKAGE = 'biglasso')
               },
               "SSR" = {
                 res <- .Call("cdfit_gaussian_ssr", X@address, yy, as.integer(row.idx-1),
                              lambda, as.integer(nlambda), as.integer(lambda.log.scale),
                              lambda.min, alpha,
                              as.integer(user.lambda | any(penalty.factor==0)),
                              eps, as.integer(max.iter), penalty.factor,
                              as.integer(dfmax), as.integer(ncores), as.integer(verbose),
                              PACKAGE = 'biglasso')
               },
               "Hybrid" = {
                 res <- .Call("cdfit_gaussian_bedpp_ssr", X@address, yy, as.integer(row.idx-1),
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
    b <- Matrix::Matrix(res[[1]], sparse = T)
    center <- res[[2]]
    scale <- res[[3]]
    lambda <- res[[4]]
    loss <- res[[5]]
    iter <- res[[6]]
    rejections <- res[[7]]
    
    if (screen %in% c("Hybrid", "Adaptive")) {
      safe_rejections <- res[[8]]
      col.idx <- res[[9]]
    } else {
      col.idx <- res[[8]]
    }
   
  } else if (family == 'binomial') {
    
    time <- system.time(
      if (alg.logistic == 'MM') {
        if(screen != "SSR") {
          warning("For now MM algorithm only supports \"SSR\" screen. Automatically switching to \"SSR\".")
          screen = "SSR"
        }
        res <- .Call("cdfit_binomial_ssr_approx", X@address, yy, as.integer(row.idx-1), 
                     lambda, as.integer(nlambda), lambda.min, alpha, 
                     as.integer(user.lambda | any(penalty.factor==0)),
                     eps, as.integer(max.iter), penalty.factor, 
                     as.integer(dfmax), as.integer(ncores), as.integer(warn),
                     as.integer(verbose),
                     PACKAGE = 'biglasso')
      } else {
        if (screen == "Hybrid") {
          res <- .Call("cdfit_binomial_slores_ssr", X@address, yy, as.integer(n.pos),
                       as.integer(ylab), as.integer(row.idx-1), 
                       lambda, as.integer(nlambda), as.integer(lambda.log.scale),
                       lambda.min, alpha, as.integer(user.lambda | any(penalty.factor==0)),
                       eps, as.integer(max.iter), penalty.factor, 
                       as.integer(dfmax), as.integer(ncores), as.integer(warn), safe.thresh,
                       as.integer(verbose),
                       PACKAGE = 'biglasso')
        }  else if(screen == "Adaptive") {
          res <- .Call("cdfit_binomial_ada_slores_ssr", X@address, yy, as.integer(n.pos),
                       as.integer(ylab), as.integer(row.idx-1), 
                       lambda, as.integer(nlambda), as.integer(lambda.log.scale),
                       lambda.min, alpha, as.integer(user.lambda | any(penalty.factor==0)),
                       eps, as.integer(max.iter), penalty.factor, 
                       as.integer(dfmax), as.integer(ncores), as.integer(warn), safe.thresh,
                       update.thresh, as.integer(verbose),
                       PACKAGE = 'biglasso')
        } else {
          res <- .Call("cdfit_binomial_ssr", X@address, yy, as.integer(row.idx-1), 
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
    b <- Matrix::Matrix(res[[2]], sparse = T)
    center <- res[[3]]
    scale <- res[[4]]
    lambda <- res[[5]]
    loss <- res[[6]]
    iter <- res[[7]]
    rejections <- res[[8]]
    
    if (screen %in% c("Hybrid", "Adaptive")) {
      safe_rejections <- res[[9]]
      col.idx <- res[[10]]
    } else {
      col.idx <- res[[9]]
    }
    
  } else if (family == "cox") {
    time <- system.time(
      if (screen == 'SSR') {
        res <- .Call("cdfit_cox_ssr", X@address, yy, d, as.integer(d_idx-1),
                     as.integer(row.idx[tOrder[row.idx.cox]]-1), lambda,
                     as.integer(nlambda), as.integer(lambda.log.scale),lambda.min,
                     alpha, as.integer(user.lambda | any(penalty.factor==0)),
                     eps, as.integer(max.iter), penalty.factor, as.integer(dfmax),
                     as.integer(ncores), as.integer(warn), as.integer(verbose),
                     PACKAGE = 'biglasso')
      } else if (screen == 'sscox') {
        res <- .Call("cdfit_cox_sscox", X@address, yy, d, as.integer(d_idx-1),
                     as.integer(row.idx[tOrder[row.idx.cox]]-1), lambda,
                     as.integer(nlambda), as.integer(lambda.log.scale),lambda.min,
                     alpha, as.integer(user.lambda | any(penalty.factor==0)),
                     eps, as.integer(max.iter), penalty.factor, as.integer(dfmax),
                     as.integer(ncores), as.integer(warn), safe.thresh, 
                     as.integer(verbose), PACKAGE = 'biglasso')
      } else if (screen == 'scox') {
        res <- .Call("cdfit_cox_scox", X@address, yy, d, as.integer(d_idx-1),
                     as.integer(row.idx[tOrder[row.idx.cox]]-1), lambda,
                     as.integer(nlambda), as.integer(lambda.log.scale),lambda.min,
                     alpha, as.integer(user.lambda | any(penalty.factor==0)),
                     eps, as.integer(max.iter), penalty.factor, as.integer(dfmax),
                     as.integer(ncores), as.integer(warn), safe.thresh, 
                     as.integer(verbose), PACKAGE = 'biglasso')
      } else if (screen == 'safe') {
        res <- .Call("cdfit_cox_safe", X@address, yy, d, as.integer(d_idx-1),
                     as.integer(row.idx[tOrder[row.idx.cox]]-1), lambda,
                     as.integer(nlambda), as.integer(lambda.log.scale),lambda.min,
                     alpha, as.integer(user.lambda | any(penalty.factor==0)),
                     eps, as.integer(max.iter), penalty.factor, as.integer(dfmax),
                     as.integer(ncores), as.integer(warn), safe.thresh, 
                     as.integer(verbose), PACKAGE = 'biglasso')
      } else if (screen == 'Adaptive') {
        res <- .Call("cdfit_cox_ada_scox", X@address, yy, d, as.integer(d_idx-1),
                     as.integer(row.idx[tOrder[row.idx.cox]]-1), lambda,
                     as.integer(nlambda), as.integer(lambda.log.scale),lambda.min,
                     alpha, as.integer(user.lambda | any(penalty.factor==0)),
                     eps, as.integer(max.iter), penalty.factor, as.integer(dfmax),
                     as.integer(ncores), as.integer(warn), safe.thresh, update.thresh,
                     as.integer(verbose), PACKAGE = 'biglasso')
      } else {
        res <- .Call("cdfit_cox", X@address, yy, d, as.integer(d_idx-1),
                     as.integer(row.idx[tOrder[row.idx.cox]]-1), lambda,
                     as.integer(nlambda), as.integer(lambda.log.scale),lambda.min,
                     alpha, as.integer(user.lambda | any(penalty.factor==0)),
                     eps, as.integer(max.iter), penalty.factor, as.integer(dfmax),
                     as.integer(ncores), as.integer(warn), as.integer(verbose),
                     PACKAGE = 'biglasso')
      }
      
    )
    
    b <- Matrix::Matrix(res[[1]], sparse = T)
    center <- res[[2]]
    scale <- res[[3]]
    lambda <- res[[4]]
    loss <- res[[5]]
    iter <- res[[6]]
    rejections <- res[[7]]
    
    if (screen %in% c("scox", "sscox", "safe")) safe_rejections <- rejections # To be updated
    if (screen %in% c("Adaptive")) {
      safe_rejections <- res[[8]]
      col.idx <- res[[9]]
    } else {
      col.idx <- res[[8]]
    }
  } else if (family == 'mgaussian') {
    time <- system.time(
      {
        switch(screen,
               "SSR" = {
                 res <- .Call("cdfit_mgaussian_ssr", X@address, yy, as.integer(row.idx-1),
                              lambda, as.integer(nlambda), as.integer(lambda.log.scale),
                              lambda.min, alpha,
                              as.integer(user.lambda | any(penalty.factor==0)),
                              eps, as.integer(max.iter), penalty.factor,
                              as.integer(dfmax), as.integer(ncores), as.integer(verbose),
                              PACKAGE = 'biglasso')
               },
               "Adaptive" = {
                 res <- .Call("cdfit_mgaussian_ada", X@address, yy, as.integer(row.idx-1),
                              lambda, as.integer(nlambda), as.integer(lambda.log.scale),
                              lambda.min, alpha,
                              as.integer(user.lambda | any(penalty.factor==0)),
                              eps, as.integer(max.iter), penalty.factor,
                              as.integer(dfmax), as.integer(ncores), 
                              safe.thresh, update.thresh, as.integer(verbose),
                              PACKAGE = 'biglasso')
               },
               stop("Invalid screening method!")
        )
      }
    )
    
    b <- res[[1]]
    center <- res[[2]]
    scale <- res[[3]]
    lambda <- res[[4]]
    loss <- res[[5]]
    iter <- res[[6]]
    rejections <- res[[7]]
    
    if (screen %in% c("Hybrid", "Adaptive")) {
      safe_rejections <- res[[8]]
      col.idx <- res[[9]]
    } else {
      col.idx <- res[[8]]
    }
    
  } else {
    stop("Current version only supports Gaussian, Binominal or Cox response!")
  }
  if (output.time) {
    cat("\nEnd biglasso: ", format(Sys.time()), '\n')
  }
  # p.keep <- length(col.idx)
  col.idx <- col.idx + 1 # indices (in R) for which variables have scale > 1e-6

  ## Eliminate saturated lambda values, if any -------------------------------
  ind <- !is.na(iter)
  if (family %in% c("gaussian","binomial")) a <- a[ind]
  if(!is.list(b)) b <- b[, ind, drop=FALSE]
  iter <- iter[ind]
  lambda <- lambda[ind]
  loss <- loss[ind]

  if (warn & any(iter==max.iter)) warning("Algorithm failed to converge for some values of lambda")

  ## Unstandardize coefficients --------------------------------------------
  if(family == "cox") {
    beta <- Matrix::Matrix(0, nrow = p, ncol = length(lambda), sparse = T)
    bb <- b / scale[col.idx]
    beta[col.idx, ] <- bb
  } else if(family == "mgaussian") {
    varnames <- if (is.null(colnames(X))) paste("V", 1:p, sep="") else colnames(X)
    varnames <- c("(Intercept)", varnames)
    a <- colMeans(y)
    nclass <- ncol(y)
    beta <- list()
    lam.idx = which(ind)
    for(class in 1:nclass) {
      beta_class <- Matrix::Matrix(0, nrow = p+1, ncol = length(lambda), sparse = T)
      beta_class[1,] <- a[class] - crossprod(center[col.idx], (b[[class]])[,ind] / scale[col.idx])
      beta_class[col.idx+1,] <- (b[[class]])[,ind] / scale[col.idx]
      #for(l in 1:length(lam.idx)) {
      #  for(j in 1:length(col.idx)) {
      #    if(b[(j-1) * nclass + class, lam.idx[l]] != 0) {
      #      beta_class[col.idx[j]+1,l] <- b[(j-1) * nclass + class, lam.idx[l]] / scale[col.idx[j]]
      #      beta_class[1,l] <- beta_class[1,l] - center[col.idx[j]] * b[(j-1) * nclass + class, lam.idx[l]] / scale[col.idx[j]]
      #    }
      #  }
      #}
      dimnames(beta_class) <- list(varnames, round(lambda, digits = 4))
      beta <- append(beta, beta_class)
    }
    yy <- t(yy)
  } else {
    beta <- Matrix::Matrix(0, nrow = (p+1), ncol = length(lambda), sparse = T)
    bb <- b / scale[col.idx]
    beta[col.idx+1, ] <- bb
    beta[1,] <- a - crossprod(center[col.idx], bb)
  }
  

  ## Names -----------------------------------------------------------
  varnames <- if (is.null(colnames(X))) paste("V", 1:p, sep="") else colnames(X)
  if(family != 'cox') varnames <- c("(Intercept)", varnames)
  if(family == "mgaussian") {
    nclass <- ncol(y)
    classnames <- if (is.null(colnames(y))) paste("class", 1:nclass, sep="") else colnames(y)
    names(beta) <- classnames
  } else dimnames(beta) <- list(varnames, round(lambda, digits = 4))

  ## Output ---------------------------------------------------------------
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
  
    if (screen %in% c("Hybrid", "Adaptive")) {
    return.val$safe_rejections <- safe_rejections
  } 
  if (return.time) return.val$time <- as.numeric(time['elapsed'])
  if(family == "mgaussian") val <- structure(return.val, class = c("mbiglasso"))
  else val <- structure(return.val, class = c("biglasso", 'ncvreg'))
  val
}
