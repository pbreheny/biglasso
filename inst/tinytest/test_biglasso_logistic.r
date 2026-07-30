if (interactive()) library(tinytest)
library(ncvreg)
library(glmnet)

# Test against glm --------------------------------------------------------

n <- 100
p <- 10
eps <- 1e-12
tolerance <- 1e-3
X <- matrix(rnorm(n*p), n, p)
# b is scaled by 1/sqrt(p) so the linear predictor's variance stays ~1
# regardless of p, keeping fitted probabilities away from the (quasi-)
# separation regime where glm()/glmnet()'s unregularized-limit MLE becomes
# numerically unstable (verified empirically: 0/2000 trials failed at this
# scale, vs. ~8% at the original unscaled b <- rnorm(p))
b <- rnorm(p, sd = 1 / sqrt(p))

y <- rbinom(n, 1, prob = exp(1 + X %*% b) / (1 + exp(1 + X %*% b)))
fit.mle <- glm(y ~ X, family = 'binomial')
beta <- fit.mle$coefficients

X.bm <- as.big.matrix(X)
fit.ssr <- biglasso(X.bm, y, family = 'binomial', eps = eps, lambda.min = 0)
fit.ssr.mm <- biglasso(X.bm, y, family = 'binomial', eps = eps, alg.logistic = 'MM', lambda.min = 0)
fit.hybrid <- biglasso(X.bm, y, family = 'binomial', eps = eps, screen = 'Hybrid', lambda.min = 0)
fit.adaptive <- biglasso(X.bm, y, family = 'binomial', eps = eps, screen = 'Adaptive', lambda.min = 0)

expect_equal(as.numeric(beta), as.numeric(fit.ssr$beta[, 100]), tolerance = tolerance)
expect_equal(as.numeric(fit.ssr$beta[, 100]), as.numeric(fit.hybrid$beta[, 100]), tolerance = tolerance)
expect_equal(as.numeric(fit.ssr$beta[, 100]), as.numeric(fit.ssr.mm$beta[, 100]), tolerance = tolerance)
expect_equal(as.numeric(fit.ssr$beta[, 100]), as.numeric(fit.adaptive$beta[, 100]), tolerance = tolerance)


# Test against glmnet -----------------------------------------------------

glmnet.control(fdev = 0, devmax = 1)
fit.glm <- glmnet(X, y, family = 'binomial', lambda.min.ratio = 0, control = list(thresh = eps))

expect_equal(as.numeric(fit.glm$beta), as.numeric(fit.ssr$beta[-1, ]), tolerance = tolerance)
expect_equal(as.numeric(fit.glm$beta), as.numeric(fit.ssr.mm$beta[-1, ]), tolerance = tolerance)
expect_equal(as.numeric(fit.glm$beta), as.numeric(fit.hybrid$beta[-1, ]), tolerance = tolerance)
expect_equal(as.numeric(fit.glm$beta), as.numeric(fit.adaptive$beta[-1, ]), tolerance = tolerance)


# Test CV against glmnet --------------------------------------------------

cv.ind <- rep(1:10, 10)

cv.default <- cv.biglasso(X.bm, y, family = 'binomial',
  eps = eps, nfolds = 10, cv.ind = cv.ind, eval.metric = "default", ncores = 1)
cv.default.ungrouped <- cv.biglasso(X.bm, y, family = 'binomial',
  eps = eps, nfolds = 10, cv.ind = cv.ind, eval.metric = "default", ncores = 1, grouped = FALSE)

cv.auc <- cv.biglasso(X.bm, y, eval.metric = "auc", family = 'binomial',
  eps = eps, nfolds = 10, cv.ind = cv.ind, ncores = 1)

cv.class <- cv.biglasso(X.bm, y, eval.metric = "class", family = 'binomial',
  eps = eps, nfolds = 10, cv.ind = cv.ind, ncores = 1)
cv.class.ungrouped <- cv.biglasso(X.bm, y, eval.metric = "class", family = 'binomial',
  eps = eps, nfolds = 10, cv.ind = cv.ind, ncores = 1, grouped = FALSE)

cv.glmnet.default <- cv.glmnet(X, y, family = 'binomial',
  lambda = cv.default$lambda, nfolds = 10, foldid = cv.ind, control = list(thresh = eps))
cv.glmnet.default.ungrouped <- cv.glmnet(X, y, family = 'binomial',
  lambda = cv.default$lambda, nfolds = 10, foldid = cv.ind, grouped = FALSE, control = list(thresh = eps))

cv.glmnet.auc <- cv.glmnet(X, y, family = 'binomial', type.measure = "auc",
  lambda = cv.default$lambda, nfolds = 10, foldid = cv.ind, control = list(thresh = eps))

cv.glmnet.class <- cv.glmnet(X, y, family = 'binomial', type.measure = "class",
  lambda = cv.default$lambda, nfolds = 10, foldid = cv.ind, control = list(thresh = eps))
cv.glmnet.class.ungrouped <- cv.glmnet(X, y, family = 'binomial', type.measure = "class",
  lambda = cv.default$lambda, nfolds = 10, foldid = cv.ind, grouped = FALSE, control = list(thresh = eps))

# default
expect_equal(cv.default$cve, cv.glmnet.default$cvm, tolerance = tolerance)
expect_equal(cv.default$cvse, cv.glmnet.default$cvsd, tolerance = tolerance)
expect_equal(cv.default.ungrouped$cve, cv.glmnet.default$cvm, tolerance = tolerance)  # comparing grouped vs. ungrouped on purpose here
expect_equal(cv.default.ungrouped$cvse, unname(cv.glmnet.default.ungrouped$cvsd), tolerance = tolerance)
expect_equal(cv.default$lambda.min, cv.glmnet.default$lambda.min)
expect_equal(cv.default.ungrouped$lambda.min, cv.glmnet.default.ungrouped$lambda.min)
expect_equal(cv.default.ungrouped$lambda.1se, cv.glmnet.default.ungrouped$lambda.1se)

# auc
expect_equal(cv.auc$cve, cv.glmnet.auc$cvm, tolerance = tolerance)
expect_equal(cv.auc$cvse, cv.glmnet.auc$cvsd, tolerance = tolerance)
expect_equal(cv.auc$lambda.min, cv.glmnet.auc$lambda.min)
expect_equal(cv.auc$lambda.1se, cv.glmnet.auc$lambda.1se)

# class
expect_equal(cv.class$cve, cv.glmnet.class$cvm, tolerance = tolerance)
expect_equal(cv.class$cvse, cv.glmnet.class$cvsd, tolerance = tolerance)
expect_equal(cv.class.ungrouped$cve, cv.glmnet.class$cvm, tolerance = tolerance)
expect_equal(cv.class.ungrouped$cvse, unname(cv.glmnet.class.ungrouped$cvsd), tolerance = tolerance)
expect_equal(cv.class$lambda.min, cv.glmnet.class$lambda.min)
expect_equal(cv.class.ungrouped$lambda.min, cv.glmnet.class.ungrouped$lambda.min)
expect_equal(cv.class.ungrouped$lambda.1se, cv.glmnet.class.ungrouped$lambda.1se)

# predictions with special lambda-values
lminpred <- predict(cv.default, X.bm, lambda = "lambda.min")
lminpred.glmnet <- predict(cv.glmnet.default, X, s = "lambda.min")
expect_equal(unname(as.matrix(lminpred)), unname(lminpred.glmnet), tolerance = tolerance)

l1sepred <- predict(cv.default, X.bm, lambda = "lambda.1se")
l1sepred.glmnet <- predict(cv.glmnet.default, X, s = "lambda.1se")
expect_equal(unname(as.matrix(l1sepred)), unname(l1sepred.glmnet), tolerance = tolerance)


# Test elastic net ----------------------------------------------------------
# compared numerically against ncvreg (same approach as the gaussian enet
# test in test_biglasso_linear.r: biglasso's enet parameterization matches
# ncvreg's exactly, unlike glmnet's).

n.en <- 200
p.en <- 30
X.en <- matrix(rnorm(n.en * p.en), n.en, p.en)
b.en <- rnorm(p.en, sd = 1 / sqrt(p.en))
y.en <- rbinom(n.en, 1, prob = 1 / (1 + exp(-(X.en %*% b.en))))
eps.en <- 1e-10
alpha.en <- 0.5

fit_ncv_en <- ncvreg(X.en, y.en, family = 'binomial', penalty = 'lasso', alpha = alpha.en,
                     eps = eps.en, lambda.min = 0.05)
X.en.bm <- as.big.matrix(X.en)
fit_big_en <- biglasso(X.en.bm, y.en, family = 'binomial', penalty = 'enet', alpha = alpha.en,
                       eps = eps.en, lambda.min = 0.05)
expect_equal(fit_big_en$screen, "SSR")  # Adaptive isn't supported for enet
expect_equal(as.numeric(fit_ncv_en$beta), as.numeric(fit_big_en$beta), tolerance = tolerance)


# Test ridge ------------------------------------------------------------------
# (Not compared numerically: penalty = "ridge" is elastic net with
# alpha = 1e-6 internally, which inflates ridge's own auto-generated lambda
# path -- see the gaussian/cox/mgaussian ridge tests for the full
# explanation. Evaluating at a lambda drawn from the lasso path's scale
# instead gives genuinely weak regularization.)

fit_lasso_en <- biglasso(X.en.bm, y.en, family = 'binomial', eps = eps.en, max.iter = 1e5)
fit_ridge_en <- biglasso(X.en.bm, y.en, family = 'binomial', penalty = 'ridge',
                         lambda = min(fit_lasso_en$lambda), eps = eps.en, max.iter = 1e5)
expect_equal(fit_ridge_en$screen, "SSR")  # Adaptive isn't supported for ridge
expect_true(mean(as.matrix(fit_ridge_en$beta[-1, , drop = FALSE]) != 0) == 1)


# Test penalty.factor -----------------------------------------------------------
# compared numerically against ncvreg (which supports the same multiplicative
# penalty.factor argument/semantics). Note penalty.factor = 0 is *not* used
# here: biglasso doesn't support unpenalized coefficients (see ?biglasso) and
# silently produces a degenerate all-zero lambda path if you try it,
# regardless of family.

pf <- rep(1, p.en)
pf[1:5] <- c(0.5, 2, 1, 3, 0.2)  # differential, all nonzero penalization
fit_ncv_pf <- ncvreg(X.en, y.en, family = 'binomial', penalty = 'lasso', eps = eps.en,
                     lambda.min = 0.05, penalty.factor = pf)
fit_big_pf <- biglasso(X.en.bm, y.en, family = 'binomial', screen = 'SSR', eps = eps.en,
                       lambda.min = 0.05, penalty.factor = pf, max.iter = 1e5)
expect_equal(fit_ncv_pf$lambda, fit_big_pf$lambda, tolerance = tolerance)
expect_equal(as.numeric(fit_ncv_pf$beta), as.numeric(fit_big_pf$beta), tolerance = tolerance)


# Test dfmax ------------------------------------------------------------------
# dfmax stops the path once the number of nonzero variables exceeds the
# bound; the lambda value that first triggers the stop is itself retained
# (matching glmnet/ncvreg convention), so it's the only point allowed to
# exceed dfmax. Tested across all screen/alg.logistic combinations, since
# each has its own C++ dfmax-early-exit code path.
#
# Regression test: screen = "Hybrid"/"Adaptive" with the default
# alg.logistic = "Newton" used to crash with "subscript out of bounds"
# whenever dfmax triggered early stopping -- the dfmax early-exit
# List::create() in cdfit_binomial_slores_ssr()/cdfit_binomial_ada_slores_ssr()
# (src/binomial.cpp) omitted the n_slores_reject element that
# R/biglasso.R's dispatch unconditionally expects at res[[10]] for these
# screens. Only alg.logistic = "MM" (which forces screen = "SSR") avoided it.

n.df <- 100
p.df <- 50
X.df <- matrix(rnorm(n.df * p.df), n.df, p.df)
b.df <- c(rnorm(20), rep(0, p.df - 20))
y.df <- rbinom(n.df, 1, prob = 1 / (1 + exp(-(X.df %*% b.df))))
X.df.bm <- as.big.matrix(X.df)

for (screen.df in c("SSR", "Hybrid", "Adaptive")) {
  fit_dfmax <- biglasso(X.df.bm, y.df, family = 'binomial', dfmax = 5, screen = screen.df,
                        eps = 1e-8, max.iter = 1e5)
  nv <- Matrix::colSums(fit_dfmax$beta[-1, , drop = FALSE] != 0)
  expect_true(length(fit_dfmax$lambda) < 100)      # path should stop early
  expect_true(all(nv[-length(nv)] <= 5))           # dfmax respected until the stopping point
  expect_true(nv[length(nv)] > 5)                  # last retained point is the one that triggered the stop
}

fit_dfmax_mm <- biglasso(X.df.bm, y.df, family = 'binomial', dfmax = 5, alg.logistic = 'MM',
                         eps = 1e-8, max.iter = 1e5)
nv_mm <- Matrix::colSums(fit_dfmax_mm$beta[-1, , drop = FALSE] != 0)
expect_true(length(fit_dfmax_mm$lambda) < 100)
expect_true(all(nv_mm[-length(nv_mm)] <= 5))
expect_true(nv_mm[length(nv_mm)] > 5)
