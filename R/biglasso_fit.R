#' COPY of biglasso, adapted to have more flexible settings 
#' 
#' NOTE: this function is designed for users who have a strong understanding of 
#' statistics and know exactly what they are doing. This is a copy of the main 
#' `biglasso()` function with more flexible settings. Of note, this function:
#'  * does NOT add an intercept 
#'  * does NOT standardize the design matrix
#'  
#'  both of the above are among the best practices for data analysis. This function 
#'  is made for use in situations where these steps have already been addressed prior 
#'  to model fitting.
#'  
#'  In other words, `biglasso_fit()` is to `biglasso()` what `ncvreg::ncvfit()`
#'  is to `ncvreg::ncvreg()`.
#'  
#'  For now, this function only works with linear regression (`family = 'gaussian'`)
#'  
#' The objective function for linear regression or multiple responses linear regression 
#' (\code{family = "gaussian"} is
#' \deqn{\frac{1}{2n}\textrm{RSS} + \lambda*\textrm{penalty},}{(1/(2n))*RSS+
#' \lambda*penalty,}
#' where for \code{family = "mgaussian"}), a group-lasso type penalty is applied.
#' @param X The design matrix, without an intercept. It must be a
#' double type \code{\link[bigmemory]{big.matrix}} object. 
#' @param y The response vector 
#' @param row.idx The integer vector of row indices of \code{X} that used for
#' fitting the model. \code{1:nrow(X)} by default.
#' @param penalty The penalty to be applied to the model. Either \code{"lasso"}
#' (the default), \code{"ridge"}, or \code{"enet"} (elastic net).
#' @param screen The feature screening rule used at each \code{lambda} that
#' discards features to speed up computation: \code{"SSR"} (default if
#' \code{penalty="ridge"} or \code{penalty="enet"} )is the sequential strong rule;
#' \code{"Adaptive"} (default for \code{penalty="lasso"}
#' without \code{penalty.factor}) is our newly proposed adaptive rules which
#' reuse screening reference for multiple lambda values. \strong{Note that:}
#' (1) for linear regression with elastic net penalty, both \code{"SSR"} and
#' \code{"Hybrid"} are applicable since version 1.3-0;  (2) only \code{"SSR"} is
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
#' each coefficient. If supplied, \code{penalty.factor} must be a numeric
#' vector of length equal to the number of columns of \code{X}.  The purpose of
#' \code{penalty.factor} is to apply differential penalization if some
#' coefficients are thought to be more likely than others to be in the model.
#' Current package doesn't allow unpenalized coefficients. That
#' is\code{penalty.factor} cannot be 0. \code{penalty.factor} is only supported
#' for "SSR" screen.
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
#' whereas the number of columns is equal to \code{nlambda}. For \code{"mgaussian"}
#' family with m responses, it is a list of m such matrices.} 
#' \item{iter}{A vector of length \code{nlambda} containing the number of 
#' iterations until convergence at each value of \code{lambda}.} 
#' \item{lambda}{The sequence of regularization parameter values in the path.}
#' \item{penalty}{Same as above.}
#' \item{family}{Same as above.}
#' \item{alpha}{Same as above.} 
#' \item{loss}{A vector containing either the residual sum of squares 
#' (for \code{"gaussian", "mgaussian"}) or negative log-likelihood
#' (for \code{"binomial", "cox"}) of the fitted model at each value of \code{lambda}.}
#' \item{penalty.factor}{Same as above.}
#' \item{n}{The number of observations used in the model fitting. It's equal to
#' \code{length(row.idx)}.} 
#' \item{center}{The sample mean vector of the variables, i.e., column mean of
#' the sub-matrix of \code{X} used for model fitting.} 
#' \item{scale}{The sample standard deviation of the variables, i.e., column
#' standard deviation of the sub-matrix of \code{X} used for model fitting.} 
#' \item{y}{The response vector used in the model fitting. Depending on
#' \code{row.idx}, it could be a subset of the raw input of the response vector y.}
#' \item{screen}{Same as above.} 
#' \item{col.idx}{The indices of features that have 'scale' value greater than
#' 1e-6. Features with 'scale' less than 1e-6 are removed from model fitting.} 
#' \item{rejections}{The number of features rejected at each value of \code{lambda}.}
#' \item{safe_rejections}{The number of features rejected by safe rules at each
#' value of \code{lambda}.}
#' @author Yaohui Zeng, Chuyi Wang, Tabitha Peter, and Patrick Breheny 
#'
#' @examples
#' 
#' data(colon)
#' X <- colon$X
#' y <- colon$y
#' X.bm <- as.big.matrix(X)
#' fit_flex <- biglasso_fit(X.bm, y)
#' plot(fit_flex, log.l = TRUE, main = 'lasso')
#'
#' # with CV 
#' 
biglasso_fit <- function(X, y, row.idx = 1:nrow(X),
                     penalty = c("lasso", "ridge", "enet"),
                     family = 'gaussian',
                     screen = "Adaptive",
                     safe.thresh = 0, update.thresh = 1, ncores = 1, alpha = 1,
                     lambda.min = ifelse(nrow(X) > ncol(X),.001,.05), 
                     nlambda = 100, lambda.log.scale = TRUE,
                     lambda, eps = 1e-7, max.iter = 1000, 
                     dfmax = ncol(X)+1,
                     penalty.factor = rep(1, ncol(X)),
                     warn = TRUE, output.time = FALSE,
                     return.time = TRUE,
                     verbose = FALSE) {
  
  
  penalty <- match.arg(penalty)
  lambda.min <- max(lambda.min, 1.0e-6)
  
  
  if (identical(penalty, "lasso")) {
    alpha <- 1
  } else if (identical(penalty, 'ridge')) {
    alpha <- 1.0e-6 ## equivalent to ridge regression
    if (screen == "Adaptive") {
      warning("For now \"ridge\" does not support \"Adaptive\" screen. Switching screen to SSR")
      screen <- 'SSR'
    }
  } else if (identical(penalty, 'enet')) {
    if (alpha >= 1 || alpha <= 0) {
      stop("alpha must be between 0 and 1 for elastic net penalty.")
    }
    if (screen == "Adaptive") {
      warning("For now \"enet\" does not support \"Adaptive\" screen. Switching screen to SSR")
      screen <- 'SSR'
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
  
  yy <- y - mean(y)
  
  p <- ncol(X)
  if (length(penalty.factor) != p) stop("penalty.factor does not match up with X")
  ## for now penalty.factor is only applicable for "SSR"
  if(any(penalty.factor != 1) & screen != "SSR") {
    warning("For now penalty.factor is only applicable for \"SSR\". Automatically switching to \"SSR\".")
    screen <- "SSR"
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
  
  ## fit model
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
               stop("Invalid screening method!")
        )
      }
    )
    
    b <- Matrix(res[[1]], sparse = T)
    center <- res[[2]]
    scale <- res[[3]]
    lambda <- res[[4]]
    loss <- res[[5]]
    iter <- res[[6]]
    rejections <- res[[7]]
    
    if (screen == "Adaptive") {
      safe_rejections <- res[[8]]
      col.idx <- res[[9]]
    } else {
      col.idx <- res[[8]]
    }
    
  } 
  
  if (output.time) {
    cat("\nEnd biglasso: ", format(Sys.time()), '\n')
  }
  
  ## Eliminate saturated lambda values, if any
  ind <- !is.na(iter)
  # if (family %in% c("gaussian","binomial")) a <- a[ind]
  if(!is.list(b)) b <- b[, ind, drop=FALSE]
  iter <- iter[ind]
  lambda <- lambda[ind]
  loss <- loss[ind]
  
  if (warn & any(iter==max.iter)) warning("Algorithm failed to converge for some values of lambda")
  
  ## Names
  varnames <- if (is.null(colnames(X))) paste("V", 1:p, sep="") else colnames(X)
  dimnames(b) <- list(varnames, round(lambda, digits = 4))
  
  ## Output
  return.val <- list(
    beta = b,
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
  
  if (screen == "Adaptive") {
    return.val$safe_rejections <- safe_rejections
  } 
  if (return.time) return.val$time <- as.numeric(time['elapsed'])
  
  val <- structure(return.val, class = c("biglasso", 'ncvreg'))
  val
}