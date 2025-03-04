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
fit_ols <- lm(y ~ X)
beta <- fit_ols$coefficients

X.bm <- as.big.matrix(X)
fit_ssr <- biglasso(X.bm, y, screen = 'SSR', eps = eps, lambda = 0)
fit_hybrid <- biglasso(X.bm, y, screen = 'Hybrid', eps = eps, lambda = 0)
fit_adaptive <- biglasso(X.bm, y, screen = 'Adaptive', eps = eps, lambda = 0)

expect_equal(as.numeric(beta), as.numeric(fit_ssr$beta), tolerance = tolerance)
expect_equal(as.numeric(beta), as.numeric(fit_hybrid$beta), tolerance = tolerance)
expect_equal(as.numeric(beta), as.numeric(fit_adaptive$beta), tolerance = tolerance)


# Test whole path against ncvreg ------------------------------------------

n <- 100
p <- 200
X <- matrix(rnorm(n*p), n, p)
b <- c(rnorm(50), rep(0, p-50))
y <- rnorm(n, X %*% b)
eps <- 1e-12
tolerance <- 1e-3
lambda.min <- 0.05

fit_ncv <- ncvreg(X, y, penalty = 'lasso', eps = eps, lambda.min = lambda.min, max.iter = 1e5)

X.bm <- as.big.matrix(X)
fit_ssr <- biglasso(X.bm, y, screen = 'SSR', eps = eps)
fit_hybrid <- biglasso(X.bm, y, screen = 'Hybrid', eps = eps)
fit_adaptive <- biglasso(X.bm, y, screen = 'Adaptive', eps = eps)

expect_equal(as.numeric(fit_ncv$beta), as.numeric(fit_ssr$beta), tolerance = tolerance)
expect_equal(as.numeric(fit_ncv$beta), as.numeric(fit_hybrid$beta), tolerance = tolerance)
expect_equal(as.numeric(fit_ncv$beta), as.numeric(fit_adaptive$beta), tolerance = tolerance)
expect_equal(fit_ncv$lambda, fit_ssr$lambda)
if (interactive()) {
  plot(fit_ncv, log.l = TRUE)
  plot(fit_ssr)
  nl <- length(fit_ncv$lambda)
  dif <- matrix(NA, nl, ncol(X) + 1)
  for (l in 1:nl) {
    dif[l, ] <- as.numeric(coef(fit_ncv, which=l) - coef(fit_ssr, which=l))
  }
  boxplot(dif)
}

# Test parallel computing -------------------------------------------------

fit_ssr2 <- biglasso(X.bm, y, screen = 'SSR', eps = eps, ncores = 2)
fit_hybrid2 <- biglasso(X.bm, y, screen = 'Hybrid', eps = eps, ncores = 2)
fit_adaptive2 <- biglasso(X.bm, y, screen = 'Adaptive', eps = eps, ncores = 2)

# Objects are mostly identical, but iterations, etc sometimes differ slightly
# expect_identical(fit_ssr, fit_ssr2)
# expect_identical(fit_hybrid, fit_hybrid2)
# expect_identical(fit_adaptive, fit_adaptive2)
expect_identical(coef(fit_ssr), coef(fit_ssr2))
expect_identical(coef(fit_hybrid), coef(fit_hybrid2))
expect_identical(coef(fit_adaptive), coef(fit_adaptive2))

# Note: biglasso has diverged from ncvreg in its approach to CV (#45)
## test_that("Test cross validation: ",{
##   expect_equal(as.numeric(cvfit_ncv$cve), as.numeric(cvfit_ssr$cve), tolerance = tolerance)
##   expect_equal(as.numeric(cvfit_ncv$cve), as.numeric(cvfit_hybrid$cve), tolerance = tolerance)
##   expect_equal(as.numeric(cvfit_ncv$cve), as.numeric(cvfit_adaptive$cve), tolerance = tolerance)
  
##   expect_equal(as.numeric(cvfit_ncv$cvse), as.numeric(cvfit_ssr$cvse), tolerance = tolerance)
##   expect_equal(as.numeric(cvfit_ncv$cvse), as.numeric(cvfit_hybrid$cvse), tolerance = tolerance)
##   expect_equal(as.numeric(cvfit_ncv$cvse), as.numeric(cvfit_adaptive$cvse), tolerance = tolerance)
  
##   expect_equal(as.numeric(cvfit_ncv$lambda.min), as.numeric(cvfit_ssr$lambda.min), tolerance = tolerance)
##   expect_equal(as.numeric(cvfit_ncv$lambda.min), as.numeric(cvfit_hybrid$lambda.min), tolerance = tolerance)
##   expect_equal(as.numeric(cvfit_ncv$lambda.min), as.numeric(cvfit_adaptive$lambda.min), tolerance = tolerance)
# cvfit_ssr <- cv.biglasso(X.bm, y, screen = 'SSR', eps = eps,
#                          ncores = 1, cv.ind = fold)
# cvfit_hybrid <- cv.biglasso(X.bm, y, screen = 'Hybrid', eps = eps,
#                             ncores = 1, cv.ind = fold)
# cvfit_adaptive <- cv.biglasso(X.bm, y, screen = 'Adaptive', eps = eps,
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

fit_ncv <- ncvreg(X, y, penalty = 'lasso', eps = sqrt(eps), 
                  lambda.min = lambda.min, alpha = alpha)
X.bm <- as.big.matrix(X)
fit_ssr <- biglasso(X.bm, y, penalty = 'enet', screen = 'SSR', eps = eps, alpha = alpha)
fit_ssr.edpp <- biglasso(X.bm, y, penalty = 'enet', screen = 'Hybrid', eps = eps, alpha = alpha)

expect_equal(as.numeric(fit_ncv$beta), as.numeric(fit_ssr$beta), tolerance = tolerance)
expect_equal(as.numeric(fit_ncv$beta), as.numeric(fit_ssr.edpp$beta), tolerance = tolerance)

## test_that("Elastic net: test cross validation: ",{
# cvfit_ncv <- cv.ncvreg(X, y, penalty = 'lasso', eps = sqrt(eps), alpha = alpha,
#                        lambda.min = lambda.min, fold = fold)
# cvfit_ssr <- cv.biglasso(X.bm, y, screen = 'SSR', penalty = 'enet', eps = eps, alpha = alpha,
#                          ncores = 1, cv.ind = fold)
# cvfit_ssr.edpp <- cv.biglasso(X.bm, y, penalty = 'enet', screen = 'Hybrid', eps = eps, alpha = alpha,
#                               ncores = 2, cv.ind = fold)
##   expect_equal(as.numeric(cvfit_ncv$cve), as.numeric(cvfit_ssr$cve), tolerance = tolerance)
##   expect_equal(as.numeric(cvfit_ncv$cve), as.numeric(cvfit_ssr.edpp$cve), tolerance = tolerance)
  
##   expect_equal(as.numeric(cvfit_ncv$cvse), as.numeric(cvfit_ssr$cvse), tolerance = tolerance)
##   expect_equal(as.numeric(cvfit_ncv$cvse), as.numeric(cvfit_ssr.edpp$cvse), tolerance = tolerance)
  
##   expect_equal(as.numeric(cvfit_ncv$lambda.min), as.numeric(cvfit_ssr$lambda.min), tolerance = tolerance)
##   expect_equal(as.numeric(cvfit_ncv$lambda.min), as.numeric(cvfit_ssr.edpp$lambda.min), tolerance = tolerance)
  
## })
