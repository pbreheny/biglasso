devtools::load_all('.')

# colon data ----------------------------------------------------------------
data(colon)
X <- colon$X |> ncvreg::std()
# X <- cbind(1, X)
xtx <- apply(X, 2, crossprod)
init <- rep(0, ncol(X)) # cold starts - use more iterations (default is 1000)
y <- colon$y
resid <- drop(y - X %*% init)
X.bm <- as.big.matrix(X)

fit1 <- biglasso_fit(X.bm, y, lambda = 0.05, xtx=xtx, r = resid, max.iter = 10000)

# compare with `ncvreg::ncvfit()`
library(ncvreg)
fit2 <- ncvfit(X = X, y = y, lambda = 0.05, xtx = xtx, r = resid, penalty = "lasso")

# test coefficients 
tinytest::expect_equal(fit1$beta, fit2$beta, tolerance = 0.01)

# test residuals
tinytest::expect_equal(fit1$resid, fit2$resid, tolerance = 0.01)

# Prostate data -----------------------------------------------
data("Prostate") # part of ncvreg
X <- Prostate$X |> ncvreg::std()
X <- cbind(1, X)
xtx <- apply(X, 2, crossprod)
init <- rep(0, ncol(X))
y <- Prostate$y
resid <- drop(y - X %*% init)
X.bm <- as.big.matrix(X)

fit3 <- biglasso_fit(X = X.bm, y = y, xtx = xtx, r = resid, lambda = 0.1,
                     max.iter = 10000)
# fit3$beta

fit4 <- ncvfit(X = X, y = y, init = init, r = resid, xtx = xtx,
               penalty = "lasso", lambda = 0.1)
# fit4$beta

tinytest::expect_equivalent(fit3$beta, fit4$beta, tolerance = 0.01)
tinytest::expect_equivalent(fit3$resid, fit4$resid, tolerance = 0.01)


## mini sim --------------------------------------------------
if (interactive()){
  
  nsim <- 100
  ncfit_res <- blfit_res <- matrix(nrow = nsim, ncol = ncol(X))
  err <- rep(NA_integer_, nsim)
  pb <- txtProgressBar(0, nsim, style = 3)
  for (i in 1:nsim){
    blfit <- biglasso_fit(X = X.bm, y = y, lambda = 0.05, xtx=xtx, r = resid)
    blfit_res[i,] <- blfit$beta
    
    ncfit <- ncvfit(X = X, y = y, lambda = 0.05, xtx = xtx, r = resid,
                    penalty = "lasso")
    ncfit_res[i,] <- ncfit$beta
    
    err[i] <- crossprod(blfit$beta - ncfit$beta)
    
    setTxtProgressBar(pb, i)
  }
  
  summary(err)
}
