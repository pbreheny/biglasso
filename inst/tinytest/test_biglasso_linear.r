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
fit_ssr <- biglasso(X.bm, y, screen = 'SSR', eps = eps, max.iter = 1e5)
fit_hybrid <- biglasso(X.bm, y, screen = 'Hybrid', eps = eps, max.iter = 1e5)
fit_adaptive <- biglasso(X.bm, y, screen = 'Adaptive', eps = eps, max.iter = 1e5)

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

fit_ssr2 <- biglasso(X.bm, y, screen = 'SSR', eps = eps, ncores = 2, max.iter = 1e5)
fit_hybrid2 <- biglasso(X.bm, y, screen = 'Hybrid', eps = eps, ncores = 2, max.iter = 1e5)
fit_adaptive2 <- biglasso(X.bm, y, screen = 'Adaptive', eps = eps, ncores = 2, max.iter = 1e5)
tol <- 1e-2

# These tests are just extremely finicky; the extent to which they agree depends on
# system architecture, the random data involved, etc. The objects tend to be *identical*,
# but sometimes they can be different, up to 0.006 differences

# expect_identical(fit_ssr, fit_ssr2)
# expect_identical(fit_hybrid, fit_hybrid2)
# expect_identical(fit_adaptive, fit_adaptive2)
expect_equivalent(coef(fit_ssr) |> as.matrix(), coef(fit_ssr2) |> as.matrix(), tolerance = tol)
expect_equivalent(coef(fit_hybrid) |> as.matrix(), coef(fit_hybrid2) |> as.matrix(), tolerance = tol)
expect_equivalent(coef(fit_adaptive) |> as.matrix(), coef(fit_adaptive2) |> as.matrix(), tolerance = tol)

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
