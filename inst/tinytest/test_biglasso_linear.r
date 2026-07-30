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


# Test ridge ----------------------------------------------------------------
# (Not compared numerically against ncvreg/glmnet: penalty = "ridge" is
# implemented internally as elastic net with alpha = 1e-6, not literal
# L2-only ridge, and lambda_max is computed as zmax / alpha. That inflates
# ridge's own auto-generated lambda path by a factor of ~1/alpha = 1e6, so
# even the smallest lambda in that path is still a very strong penalty in
# absolute terms -- not the weakly regularized regime a density check needs.
# Evaluating at a lambda drawn from the lasso path's scale instead gives
# genuinely weak regularization; same approach as the cox/mgaussian ridge
# tests.)

fit_lasso <- biglasso(X.bm, y, eps = eps, max.iter = 1e5)
fit_ridge <- biglasso(X.bm, y, penalty = 'ridge', lambda = min(fit_lasso$lambda),
                      eps = eps, max.iter = 1e5)
expect_equal(fit_ridge$screen, "SSR")  # Adaptive isn't supported for ridge
expect_true(mean(as.matrix(fit_ridge$beta[-1, , drop = FALSE]) != 0) == 1)


# Test penalty.factor ---------------------------------------------------------
# compared numerically against ncvreg, which supports the same multiplicative
# penalty.factor argument/semantics. Uses a smaller, well-conditioned (n > p)
# design than the enet test above, since with n < p even tiny numerical
# differences between the two implementations get amplified into a
# substantial max-abs-difference in beta. Note penalty.factor = 0 is *not*
# used here: biglasso doesn't support unpenalized coefficients (see
# ?biglasso) and silently produces a degenerate all-zero lambda path if you
# try it, regardless of family.

n.pf <- 200
p.pf <- 30
X.pf <- matrix(rnorm(n.pf * p.pf), n.pf, p.pf)
b.pf <- c(rnorm(10), rep(0, p.pf - 10))
y.pf <- rnorm(n.pf, X.pf %*% b.pf)
X.pf.bm <- as.big.matrix(X.pf)
eps.pf <- 1e-10

pf <- rep(1, p.pf)
pf[1:5] <- c(0.5, 2, 1, 3, 0.2)  # differential, all nonzero penalization
fit_ncv_pf <- ncvreg(X.pf, y.pf, penalty = 'lasso', eps = eps.pf, lambda.min = 0.05,
                     penalty.factor = pf)
fit_big_pf <- biglasso(X.pf.bm, y.pf, screen = 'SSR', eps = eps.pf, lambda.min = 0.05,
                       penalty.factor = pf, max.iter = 1e5)
expect_equal(fit_ncv_pf$lambda, fit_big_pf$lambda, tolerance = tolerance)
expect_equal(as.numeric(fit_ncv_pf$beta), as.numeric(fit_big_pf$beta), tolerance = tolerance)


# Test dfmax ------------------------------------------------------------------
# dfmax stops the path once the number of nonzero variables exceeds the
# bound; the lambda value that first triggers the stop is itself retained
# (matching glmnet/ncvreg convention), so it's the only point allowed to
# exceed dfmax. Tested across all three screening rules, since each has its
# own C++ dfmax-early-exit code path (see the analogous cox/mgaussian dfmax
# tests, and the binomial dfmax test in test_biglasso_logistic.r, which used
# to crash for Hybrid/Adaptive).

for (screen.dfmax in c("SSR", "Hybrid", "Adaptive")) {
  fit_dfmax <- biglasso(X.bm, y, dfmax = 5, screen = screen.dfmax, eps = eps, max.iter = 1e5)
  nv <- Matrix::colSums(fit_dfmax$beta[-1, , drop = FALSE] != 0)
  expect_true(length(fit_dfmax$lambda) < 100)      # path should stop early
  expect_true(all(nv[-length(nv)] <= 5))           # dfmax respected until the stopping point
  expect_true(nv[length(nv)] > 5)                  # last retained point is the one that triggered the stop
}
