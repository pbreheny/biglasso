#' Model predictions based on a fitted \code{biglasso} object
#' 
#' Extract predictions (fitted reponse, coefficients, etc.) from a 
#' fitted \code{\link{biglasso}} object.
#' 
#' @name predict.biglasso
#' @rdname predict.biglasso
#' @method predict biglasso
#' 
#' @param object A fitted \code{"biglasso"} model object.
#' @param X Matrix of values at which predictions are to be made. It must be a
#' \code{\link[bigmemory]{big.matrix}} object. Not used for
#' \code{type="coefficients"}.
#' @param row.idx Similar to that in \code{\link[biglasso]{biglasso}}, it's a
#' vector of the row indices of \code{X} that used for the prediction.
#' \code{1:nrow(X)} by default.
#' @param type Type of prediction: \code{"link"} returns the linear predictors;
#' \code{"response"} gives the fitted values; \code{"class"} returns the
#' binomial outcome with the highest probability; \code{"coefficients"} returns
#' the coefficients; \code{"vars"} returns a list containing the indices and
#' names of the nonzero variables at each value of \code{lambda};
#' \code{"nvars"} returns the number of nonzero coefficients at each value of
#' \code{lambda}.
#' @param lambda Values of the regularization parameter \code{lambda} at which
#' predictions are requested.  Linear interpolation is used for values of
#' \code{lambda} not in the sequence of lambda values in the fitted models.
#' @param k Index of the response to predict in multiple responses regression (
#' \code{family="mgaussian"}).
#' @param which Indices of the penalty parameter \code{lambda} at which
#' predictions are required.  By default, all indices are returned.  If
#' \code{lambda} is specified, this will override \code{which}.
#' @param intercept Whether the intercept should be included in the returned
#' coefficients. For \code{family="mgaussian"} only.
#' @param drop If coefficients for a single value of \code{lambda} are to be
#' returned, reduce dimensions to a vector?  Setting \code{drop=FALSE} returns
#' a 1-column matrix.
#' @param \dots Not used.
#' @return The object returned depends on \code{type}.
#' @author Yaohui Zeng and Patrick Breheny
#' 
#' Maintainer: Yaohui Zeng <yaohui.zeng@@gmail.com>
#' @seealso \code{\link{biglasso}}, \code{\link{cv.biglasso}}
#' @keywords models regression
#' @examples
#' ## Logistic regression
#' data(colon)
#' X <- colon$X
#' y <- colon$y
#' X.bm <- as.big.matrix(X, backingfile = "")
#' fit <- biglasso(X.bm, y, penalty = 'lasso', family = "binomial")
#' coef <- coef(fit, lambda=0.05, drop = TRUE)
#' coef[which(coef != 0)]
#' predict(fit, X.bm, type="link", lambda=0.05)[1:10]
#' predict(fit, X.bm, type="response", lambda=0.05)[1:10]
#' predict(fit, X.bm, type="class", lambda=0.1)[1:10]
#' predict(fit, type="vars", lambda=c(0.05, 0.1))
#' predict(fit, type="nvars", lambda=c(0.05, 0.1))
#' @export
#' 
predict.biglasso <- function(object, X, row.idx = 1:nrow(X), 
                             type = c("link", "response", "class", 
                                    "coefficients", "vars", "nvars"),
                             lambda, which = 1:length(object$lambda), ...) {
  type <- match.arg(type)
  beta <- coef.biglasso(object, lambda=lambda, which=which, drop=FALSE)
  if (type=="coefficients") return(beta)
  if (class(object)[1]=="biglasso" && object$family != "cox") {
    alpha <- beta[1,]
    beta <- beta[-1,,drop=FALSE]
  }
  
  if (type=="nvars") return(apply(beta!=0,2,sum, na.rm = T))
  if (type=="vars") return(drop(apply(beta!=0, 2, FUN=which)))

  if (!inherits(X, 'big.matrix')) {
    stop("X must be a big.matrix object.")
  }
 
  beta.T <- as(beta, "TsparseMatrix") 
  temp <- get_eta(X@address, as.integer(row.idx-1), beta, beta.T@i, beta.T@j)
  if(object$family != "cox") eta <- sweep(temp, 2, alpha, "+")
  # dimnames(eta) <- list(c(1:nrow(eta)), round(object$lambda, digits = 4))
  
  if (object$family != 'binomial') {
    if (type == 'class') {
      stop("type='class' can only be used with family='binomial'")
    } else { ## then 'type' must be either 'link' or 'response'
      return(drop(eta))
    }
  } else { # binomial
    if (type =='link') {
      return(drop(eta))
    } else if (type == 'class') {
      return(drop(Matrix(1*(eta>0))))
    } else { # response
      return(drop(exp(eta)/(1+exp(eta))))
    }
  }
}

#' @method predict mbiglasso
#' @rdname predict.biglasso
#' @export
#'
predict.mbiglasso <- function(object, X, row.idx = 1:nrow(X), 
                             type = c("link", "response", 
                                      "coefficients", "vars", "nvars"),
                             lambda, which = 1:length(object$lambda), k = 1, ...) {
  type <- match.arg(type)
  beta <- coef.biglasso(object, lambda=lambda, which=which, drop=FALSE)[[k]]
  if (type=="coefficients") return(beta)
  if (class(object)[1]=="biglasso") {
    alpha <- beta[1,]
    beta <- beta[-1,,drop=FALSE]
  }
  
  if (type=="nvars") return(apply(beta!=0,2,sum, na.rm = T))
  if (type=="vars") return(drop(apply(beta!=0, 2, FUN=which)))
  
  if (!inherits(X, 'big.matrix')) {
    stop("X must be a big.matrix object.")
  }
  
  beta.T <- as(beta, "TsparseMatrix") 
  temp <- get_eta(X@address, as.integer(row.idx-1), beta, beta.T@i, beta.T@j)
  eta <- sweep(temp, 2, alpha, "+")
  # dimnames(eta) <- list(c(1:nrow(eta)), round(object$lambda, digits = 4))
  
  return(eta)
}

#' @method coef biglasso
#' @rdname predict.biglasso
#' @export
#'
coef.biglasso <- function(object, lambda, which = 1:length(object$lambda), drop = TRUE, ...) {
  if (!missing(lambda)) {
    ind <- approx(object$lambda,seq(object$lambda),lambda)$y
    l <- floor(ind)
    r <- ceiling(ind)
    w <- ind %% 1
    beta <- (1-w)*object$beta[,l,drop=FALSE] + w*object$beta[,r,drop=FALSE]
    colnames(beta) <- round(lambda,4)
  }
  else beta <- object$beta[,which,drop=FALSE]
  if (drop) return(drop(beta)) else return(beta)
}

#' @method coef mbiglasso
#' @rdname predict.biglasso
#' @export
#'
coef.mbiglasso <- function(object, lambda, which = 1:length(object$lambda), intercept = TRUE, ...) {
  nclass = length(object$beta)
  beta = list()
  if(intercept) col.idx = 1:nrow(object$beta[[1]])
  else col.idx = 2:nrow(object$beta[[1]])
  if (!missing(lambda)) {
    ind <- approx(object$lambda,seq(object$lambda),lambda)$y
    l <- floor(ind)
    r <- ceiling(ind)
    w <- ind %% 1
    for(class in 1:nclass) {
      beta_class = (1-w)*(object$beta[[class]])[col.idx,l,drop=FALSE] + w*(object$beta[[class]])[col.idx,r,drop=FALSE]
      colnames(beta_class) <- round(lambda,4)
      beta = append(beta, beta_class)
    }
  }
  else{
    for(class in 1:nclass) {
      beta_class = (object$beta[[class]])[col.idx,which,drop=FALSE]
      beta = append(beta, beta_class)
    }
  } 
  return(beta)
}