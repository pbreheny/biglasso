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

## lasso ---------------------------------------------------------------------
fit1 <- biglasso_fit(X.bm, y, lambda = 0.05, xtx=xtx, r = resid, max.iter = 10000)

# compare with `ncvreg::ncvfit()`
library(ncvreg)
fit2 <- ncvfit(X = X, y = y, lambda = 0.05, xtx = xtx, r = resid, penalty = "lasso")

# test coefficients 
tinytest::expect_equal(fit1$beta, fit2$beta, tolerance = 0.01)

# test residuals
tinytest::expect_equal(fit1$resid, fit2$resid, tolerance = 0.01)

## MCP --------------------------------------------------------------
fit1b <- biglasso_fit(X.bm, y, lambda = 0.05,
                              xtx=xtx, r = resid,
                              penalty = "MCP")

fit2b <- ncvfit(X = X, y = y, lambda = 0.05, xtx = xtx, r = resid,
                penalty = "MCP")

tinytest::expect_equal(fit1b$resid, fit2b$resid, tolerance = 0.01)


## SCAD + path (just an idea) -----------------------------------------------------------
X_plus_int <- cbind(1, X)
X.bm.int <- as.big.matrix(X_plus_int)
xtx.int <- apply(X_plus_int, 2, crossprod)
resid.int <- drop(y - X_plus_int %*% c(0, init))

fit1c <- biglasso_simple_path(X.bm.int, y, lambda = c(0.1, 0.05, 0.01, 0.001),
                              xtx=xtx.int, r = resid.int,
                              penalty = "MCP",
                              # make intercept unpenalized 
                              penalty.factor = c(0, rep(1, ncol(X))),
                              max.iter = 20000)

fit2c <- ncvreg(X, y, lambda = c(0.1, 0.05, 0.01, 0.001),
                max.iter = 10000)

# TODO: think about whether I can create a test like the below
# tinytest::expect_equal(fit1c$beta[,2], fit2c$beta[,2], tolerance = 0.01)

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
               penalty = "lasso", lambda = 0.1)
# fit4$beta
tinytest::expect_equivalent(fit3$beta, fit4$beta, tolerance = 0.01)
tinytest::expect_equivalent(fit3$resid, fit4$resid, tolerance = 0.01)

## MCP ---------------------------------------------------------------------
fit3b <- biglasso_fit(X = X.bm, y = y, xtx = xtx, r = resid, lambda = 0.1,
                     penalty = "MCP",
                     max.iter = 10000)

fit4b <- ncvfit(X = X, y = y, init = init, r = resid, xtx = xtx,
               penalty = "MCP", lambda = 0.1)

tinytest::expect_equivalent(fit3b$resid, fit4b$resid, tolerance = 0.01)

## SCAD --------------------------------------------------------------------

fit3c <- biglasso_fit(X = X.bm, y = y, xtx = xtx, r = resid, lambda = 0.1,
                      penalty = "SCAD",
                      max.iter = 10000)

fit4c <- ncvfit(X = X, y = y, init = init, r = resid, xtx = xtx,
                penalty = "SCAD", lambda = 0.1)

tinytest::expect_equivalent(fit3c$resid, fit4c$resid, tolerance = 0.01)

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
