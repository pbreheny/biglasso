if (interactive()) library(tinytest)
library(ncvreg)
library(glmnet)

# Test against OLS --------------------------------------------------------

n <- 100
p <- 10
eps <- 1e-10
tolerance <- 1e-4
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


# Test whole path against ncvreg ------------------------------------------

n <- 100
p <- 200
X <- matrix(rnorm(n*p), n, p)
b <- c(rnorm(50), rep(0, p-50))
y <- rnorm(n, X %*% b)
eps <- 1e-12
tolerance <- 1e-3
lambda.min <- 0.05

fit.ncv <- ncvreg(X, y, penalty = 'lasso', eps = eps, lambda.min = lambda.min)

X.bm <- as.big.matrix(X)
fit.ssr <- biglasso(X.bm, y, screen = 'SSR', eps = eps)
fit.hybrid <- biglasso(X.bm, y, screen = 'Hybrid', eps = eps)
fit.adaptive <- biglasso(X.bm, y, screen = 'Adaptive', eps = eps)

expect_equal(as.numeric(fit.ncv$beta), as.numeric(fit.ssr$beta), tolerance = tolerance)
expect_equal(as.numeric(fit.ncv$beta), as.numeric(fit.hybrid$beta), tolerance = tolerance)
expect_equal(as.numeric(fit.ncv$beta), as.numeric(fit.adaptive$beta), tolerance = tolerance)
expect_equal(fit.ncv$lambda, fit.ssr$lambda)
if (interactive()) {
  plot(fit.ncv, log.l = TRUE)
  plot(fit.ssr)
  nl <- length(fit.ncv$lambda)
  dif <- matrix(NA, nl, ncol(X) + 1)
  for (l in 1:nl) {
    dif[l, ] <- as.numeric(coef(fit.ncv, which=l) - coef(fit.ssr, which=l))
  }
  boxplot(dif)
}

# Test parallel computing -------------------------------------------------

fit.ssr2 <- biglasso(X.bm, y, screen = 'SSR', eps = eps, ncores = 2)
fit.hybrid2 <- biglasso(X.bm, y, screen = 'Hybrid', eps = eps, ncores = 2)
fit.adaptive2 <- biglasso(X.bm, y, screen = 'Adaptive', eps = eps, ncores = 2)

fit.ssr$time <- NA
fit.ssr2$time <- NA
fit.hybrid$time <- NA
fit.hybrid2$time <- NA
fit.adaptive$time <- NA
fit.adaptive2$time <- NA
fit.hybrid$safe_rejections <- NA   # These are usually very similar, but
fit.hybrid2$safe_rejections <- NA  # not necessarily identical
expect_identical(fit.ssr, fit.ssr2)
expect_identical(fit.hybrid, fit.hybrid2)
expect_identical(fit.adaptive, fit.adaptive2)

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
# cvfit.ssr <- cv.biglasso(X.bm, y, screen = 'SSR', eps = eps,
#                          ncores = 1, cv.ind = fold)
# cvfit.hybrid <- cv.biglasso(X.bm, y, screen = 'Hybrid', eps = eps,
#                             ncores = 1, cv.ind = fold)
# cvfit.adaptive <- cv.biglasso(X.bm, y, screen = 'Adaptive', eps = eps,
#                               ncores = 1, cv.ind = fold)


## })


# Test elastic net --------------------------------------------------------

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

expect_equal(as.numeric(fit.ncv$beta), as.numeric(fit.ssr$beta), tolerance = tolerance)
expect_equal(as.numeric(fit.ncv$beta), as.numeric(fit.ssr.edpp$beta), tolerance = tolerance)

## test_that("Elastic net: test cross validation: ",{
# cvfit.ncv <- cv.ncvreg(X, y, penalty = 'lasso', eps = sqrt(eps), alpha = alpha,
#                        lambda.min = lambda.min, fold = fold)
# cvfit.ssr <- cv.biglasso(X.bm, y, screen = 'SSR', penalty = 'enet', eps = eps, alpha = alpha,
#                          ncores = 1, cv.ind = fold)
# cvfit.ssr.edpp <- cv.biglasso(X.bm, y, penalty = 'enet', screen = 'Hybrid', eps = eps, alpha = alpha,
#                               ncores = 2, cv.ind = fold)
##   expect_equal(as.numeric(cvfit.ncv$cve), as.numeric(cvfit.ssr$cve), tolerance = tolerance)
##   expect_equal(as.numeric(cvfit.ncv$cve), as.numeric(cvfit.ssr.edpp$cve), tolerance = tolerance)
  
##   expect_equal(as.numeric(cvfit.ncv$cvse), as.numeric(cvfit.ssr$cvse), tolerance = tolerance)
##   expect_equal(as.numeric(cvfit.ncv$cvse), as.numeric(cvfit.ssr.edpp$cvse), tolerance = tolerance)
  
##   expect_equal(as.numeric(cvfit.ncv$lambda.min), as.numeric(cvfit.ssr$lambda.min), tolerance = tolerance)
##   expect_equal(as.numeric(cvfit.ncv$lambda.min), as.numeric(cvfit.ssr.edpp$lambda.min), tolerance = tolerance)
  
## })
