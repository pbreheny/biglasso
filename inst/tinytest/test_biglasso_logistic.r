if (interactive()) library(tinytest)
library(ncvreg)
library(glmnet)


# Test against glm --------------------------------------------------------

n <- 100
p <- 10
eps <- 1e-12
tolerance <- 1e-3
X <- matrix(rnorm(n*p), n, p)
b <- rnorm(p)

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
fit.glm <- glmnet(X, y, family = 'binomial', thresh = eps, lambda.min.ratio = 0)

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
  lambda = cv.default$lambda, nfolds = 10, foldid = cv.ind, thresh = eps)
cv.glmnet.default.ungrouped <- cv.glmnet(X, y, family = 'binomial',
  lambda = cv.default$lambda, nfolds = 10, foldid = cv.ind, thresh = eps, grouped = FALSE)

cv.glmnet.auc <- cv.glmnet(X, y, family = 'binomial', type.measure = "auc",
  lambda = cv.default$lambda, nfolds = 10, foldid = cv.ind, thresh = eps)

cv.glmnet.class <- cv.glmnet(X, y, family = 'binomial', type.measure = "class",
  lambda = cv.default$lambda, nfolds = 10, foldid = cv.ind, thresh = eps)
cv.glmnet.class.ungrouped <- cv.glmnet(X, y, family = 'binomial', type.measure = "class",
  lambda = cv.default$lambda, nfolds = 10, foldid = cv.ind, thresh = eps, grouped = FALSE)

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
