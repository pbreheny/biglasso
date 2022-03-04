
library(testthat)
library(biglasso)
library(ncvreg)
library(glmnet)

context("Testing linear regression:")

test_that("Test against OLS:", {
  ## test against OLS
  set.seed(1234)
  n <- 100
  p <- 10
  eps <- 1e-10
  tolerance <- 1e-6
  X <- matrix(rnorm(n*p), n, p)
  b <- rnorm(p)
  y <- rnorm(n, X %*% b)
  fit.ols <- lm(y ~ X)
  beta <- fit.ols$coefficients
  
  X.bm <- as.big.matrix(X)
  fit.ssr <- biglasso(X.bm, y, screen = 'SSR', eps = eps, lambda = 0)
  fit.hybrid <- biglasso(X.bm, y, screen = 'Hybrid', eps = eps, lambda = 0)
  fit.adaptive <- biglasso(X.bm, y, screen = 'Adaptive', eps = eps, lambda = 0)
  
  expect_equal(as.numeric(beta), as.numeric(fit.ssr$beta), tolerance = tolerance)
  expect_equal(as.numeric(beta), as.numeric(fit.hybrid$beta), tolerance = tolerance)
  expect_equal(as.numeric(beta), as.numeric(fit.adaptive$beta), tolerance = tolerance)

})

set.seed(1234)
n <- 100
p <- 200
X <- matrix(rnorm(n*p), n, p)
b <- c(rnorm(50), rep(0, p-50))
y <- rnorm(n, X %*% b)
eps <- 1e-8
tolerance <- 1e-3
lambda.min <- 0.05
fold = sample(rep(1:5, length.out = n))

fit.ncv <- ncvreg(X, y, penalty = 'lasso', eps = sqrt(eps), lambda.min = lambda.min)
cvfit.ncv <- cv.ncvreg(X, y, penalty = 'lasso', eps = sqrt(eps), 
                       lambda.min = lambda.min, fold = fold)

X.bm <- as.big.matrix(X)
fit.ssr <- biglasso(X.bm, y, screen = 'SSR', eps = eps)
fit.hybrid <- biglasso(X.bm, y, screen = 'Hybrid', eps = eps)
fit.adaptive <- biglasso(X.bm, y, screen = 'Adaptive', eps = eps)


cvfit.ssr <- cv.biglasso(X.bm, y, screen = 'SSR', eps = eps,
                         ncores = 1, cv.ind = fold)
cvfit.hybrid <- cv.biglasso(X.bm, y, screen = 'Hybrid', eps = eps,
                              ncores = 1, cv.ind = fold)
cvfit.adaptive <- cv.biglasso(X.bm, y, screen = 'Adaptive', eps = eps,
                              ncores = 1, cv.ind = fold)

## parallel computing
# fit.edpp.no.active2 <- biglasso(X.bm, y, screen = 'SEDPP-No-Active', eps = eps, ncores = 2)
fit.ssr2 <- biglasso(X.bm, y, screen = 'SSR', eps = eps, ncores = 2)
fit.hybrid2 <- biglasso(X.bm, y, screen = 'Hybrid', eps = eps, ncores = 2)
fit.adaptive2 <- biglasso(X.bm, y, screen = 'Adaptive', eps = eps, ncores = 2)

test_that("Test against ncvreg for entire path:", {
  expect_equal(as.numeric(fit.ncv$beta), as.numeric(fit.ssr$beta), tolerance = tolerance)
  expect_equal(as.numeric(fit.ncv$beta), as.numeric(fit.hybrid$beta), tolerance = tolerance)
  expect_equal(as.numeric(fit.ncv$beta), as.numeric(fit.adaptive$beta), tolerance = tolerance)
})

test_that("Test parallel computing: ",{
  fit.ssr$time <- NA
  fit.ssr2$time <- NA
  fit.hybrid$time <- NA
  fit.hybrid2$time <- NA
  fit.adaptive$time <- NA
  fit.adaptive2$time <- NA
  expect_identical(fit.ssr, fit.ssr2)
  expect_identical(fit.hybrid, fit.hybrid2)
  expect_identical(fit.adaptive, fit.adaptive2)
})

# Note: biglasso has diverged from ncvreg in its approach to CV (#45)
## test_that("Test cross validation: ",{
##   expect_equal(as.numeric(cvfit.ncv$cve), as.numeric(cvfit.ssr$cve), tolerance = tolerance)
##   expect_equal(as.numeric(cvfit.ncv$cve), as.numeric(cvfit.hybrid$cve), tolerance = tolerance)
##   expect_equal(as.numeric(cvfit.ncv$cve), as.numeric(cvfit.adaptive$cve), tolerance = tolerance)
  
##   expect_equal(as.numeric(cvfit.ncv$cvse), as.numeric(cvfit.ssr$cvse), tolerance = tolerance)
##   expect_equal(as.numeric(cvfit.ncv$cvse), as.numeric(cvfit.hybrid$cvse), tolerance = tolerance)
##   expect_equal(as.numeric(cvfit.ncv$cvse), as.numeric(cvfit.adaptive$cvse), tolerance = tolerance)
  
##   expect_equal(as.numeric(cvfit.ncv$lambda.min), as.numeric(cvfit.ssr$lambda.min), tolerance = tolerance)
##   expect_equal(as.numeric(cvfit.ncv$lambda.min), as.numeric(cvfit.hybrid$lambda.min), tolerance = tolerance)
##   expect_equal(as.numeric(cvfit.ncv$lambda.min), as.numeric(cvfit.adaptive$lambda.min), tolerance = tolerance)
  
## })

# ------------------------------------------------------------------------------
# test elastic net
# ------------------------------------------------------------------------------
set.seed(1234)
n <- 100
p <- 200
X <- matrix(rnorm(n*p), n, p)
b <- c(rnorm(50), rep(0, p-50))
y <- rnorm(n, X %*% b)
eps <- 1e-8
tolerance <- 1e-3
lambda.min <- 0.05
alpha <- 0.5
fold = sample(rep(1:5, length.out = n))

fit.ncv <- ncvreg(X, y, penalty = 'lasso', eps = sqrt(eps), 
                  lambda.min = lambda.min, alpha = alpha)
X.bm <- as.big.matrix(X)
fit.ssr <- biglasso(X.bm, y, penalty = 'enet', screen = 'SSR', eps = eps, alpha = alpha)
fit.ssr.edpp <- biglasso(X.bm, y, penalty = 'enet', screen = 'Hybrid', eps = eps, alpha = alpha)

cvfit.ncv <- cv.ncvreg(X, y, penalty = 'lasso', eps = sqrt(eps), alpha = alpha,
                       lambda.min = lambda.min, fold = fold)
cvfit.ssr <- cv.biglasso(X.bm, y, screen = 'SSR', penalty = 'enet', eps = eps, alpha = alpha,
                         ncores = 1, cv.ind = fold)
cvfit.ssr.edpp <- cv.biglasso(X.bm, y, penalty = 'enet', screen = 'Hybrid', eps = eps, alpha = alpha,
                              ncores = 2, cv.ind = fold)

test_that("Elastic net: test against ncvreg for entire path:", {
  expect_equal(as.numeric(fit.ncv$beta), as.numeric(fit.ssr$beta), tolerance = tolerance)
  expect_equal(as.numeric(fit.ncv$beta), as.numeric(fit.ssr.edpp$beta), tolerance = tolerance)
})

## test_that("Elastic net: test cross validation: ",{
##   expect_equal(as.numeric(cvfit.ncv$cve), as.numeric(cvfit.ssr$cve), tolerance = tolerance)
##   expect_equal(as.numeric(cvfit.ncv$cve), as.numeric(cvfit.ssr.edpp$cve), tolerance = tolerance)
  
##   expect_equal(as.numeric(cvfit.ncv$cvse), as.numeric(cvfit.ssr$cvse), tolerance = tolerance)
##   expect_equal(as.numeric(cvfit.ncv$cvse), as.numeric(cvfit.ssr.edpp$cvse), tolerance = tolerance)
  
##   expect_equal(as.numeric(cvfit.ncv$lambda.min), as.numeric(cvfit.ssr$lambda.min), tolerance = tolerance)
##   expect_equal(as.numeric(cvfit.ncv$lambda.min), as.numeric(cvfit.ssr.edpp$lambda.min), tolerance = tolerance)
  
## })

