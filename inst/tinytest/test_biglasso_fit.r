library(ncvreg)
library(glmnet)

# colon data ----------------------------------------------------------------
data(colon)
X <- colon$X |> ncvreg::std()
# X <- cbind(1, X)
xtx <- apply(X, 2, crossprod)
init <- rep(0, ncol(X)) # cold starts - use more iterations (default is 1000)
y <- colon$y
og_resid <- resid <- drop(y - X %*% init)
og_X <- X.bm <- as.big.matrix(X)

## lasso ---------------------------------------------------------------------
fit1 <- biglasso_fit(X.bm, y, lambda = 0.05, xtx=xtx, r = resid,
                     penalty = "lasso", max.iter = 10000)

# compare with `ncvreg::ncvfit()`
fit2 <- ncvfit(X = X, y = y, lambda = 0.05, xtx = xtx, r = resid,
               penalty = "lasso", max.iter = 10000)

# test coefficients 
expect_equivalent(fit1$beta, fit2$beta, tolerance = 0.001)

# test residuals
expect_equivalent(fit1$resid, fit2$resid, tolerance = 0.001)

## MCP --------------------------------------------------------------
fit1b <- biglasso_fit(X.bm, y, lambda = 0.05, xtx=xtx, r = resid,
                              penalty = "MCP", max.iter = 10000)

fit2b <- ncvfit(X = X, y = y, lambda = 0.05, xtx = xtx, r = resid,
                penalty = "MCP", max.iter = 10000)

expect_equivalent(fit1b$resid, fit2b$resid, tolerance = 0.01)
expect_equivalent(fit1b$beta, fit2b$beta, tolerance = 0.01)

# SCAD --------------------------------------------------------------
fit1c <- biglasso_fit(X.bm, y, lambda = 0.05, xtx=xtx, r = resid,
                      penalty = "SCAD", max.iter = 10000)

fit2c <- ncvfit(X = X, y = y, lambda = 0.05, xtx = xtx, r = resid,
                penalty = "SCAD", max.iter = 10000)


expect_equivalent(fit1c$resid, fit2c$resid, tolerance = 0.01)
expect_equivalent(fit1c$beta, fit2c$beta, tolerance = 0.01)

# Prostate data ------------------------------------------------------------
data("Prostate") # part of ncvreg
X <- Prostate$X |> ncvreg::std()
X <- cbind(1, X)
xtx <- apply(X, 2, crossprod)
init <- rep(0, ncol(X))
y <- Prostate$y
resid <- drop(y - X %*% init)
X.bm <- as.big.matrix(X)

## lasso ------------------------------------------------------------
fit3 <- biglasso_fit(X = X.bm, y = y, xtx = xtx, r = resid, lambda = 0.1,
                     penalty = "lasso",
                     max.iter = 10000)
# fit3$beta

fit4 <- ncvfit(X = X, y = y, init = init, r = resid, xtx = xtx,
               penalty = "lasso", lambda = 0.1, max.iter = 10000)
# fit4$beta
expect_equivalent(fit3$beta, fit4$beta, tolerance = 0.01)
expect_equivalent(fit3$resid, fit4$resid, tolerance = 0.01)

## MCP ---------------------------------------------------------------------
fit3b <- biglasso_fit(X = X.bm, y = y, xtx = xtx, r = resid, lambda = 0.1,
                     penalty = "MCP",
                     max.iter = 10000)

fit4b <- ncvfit(X = X, y = y, init = init, r = resid, xtx = xtx,
               penalty = "MCP", lambda = 0.1, max.iter = 10000)

expect_equivalent(fit3b$resid, fit4b$resid, tolerance = 0.01)
expect_equivalent(fit3b$beta, fit4b$beta, tolerance = 0.01)

## SCAD --------------------------------------------------------------------
fit3c <- biglasso_fit(X = X.bm, y = y, xtx = xtx, r = resid, lambda = 0.1,
                      penalty = "SCAD",
                      max.iter = 10000)

fit4c <- ncvfit(X = X, y = y, init = init, r = resid, xtx = xtx,
                penalty = "SCAD", lambda = 0.1, max.iter = 10000)

expect_equivalent(fit3c$resid, fit4c$resid, tolerance = 0.01)
expect_equivalent(fit3c$beta, fit4c$beta, tolerance = 0.01)

# mini sim --------------------------------------------------
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
