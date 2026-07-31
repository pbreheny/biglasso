if (interactive()) {
  library(tinytest)
}
library(glmnet)

# NOTE on comparing against glmnet: biglasso's multi-response ("mgaussian")
# group-lasso update divides the joint (across-response) norm by sqrt(m)
# (see standardize_and_get_residual()/lasso() in src/mgaussian.cpp), so its
# lambda sequence is glmnet's divided by sqrt(m). Accounting for that
# rescaling, the lasso (alpha = 1) solutions from the two packages agree
# essentially exactly.

# Test whole path against glmnet (lasso) -----------------------------------

n <- 150
p <- 60
m <- 3
eps <- 1e-12
tolerance <- 1e-3
X <- matrix(rnorm(n * p), n, p)
B <- matrix(0, p, m)
B[1:6, ] <- matrix(rnorm(6 * m), 6, m)
Y <- X %*% B + matrix(rnorm(n * m, sd = 0.5), n, m)

X.bm <- as.big.matrix(X)
fit_glmnet <- glmnet(X, Y, family = "mgaussian", control = list(thresh = eps))
lambda_big <- fit_glmnet$lambda / sqrt(m)

fit_ssr <- biglasso(
  X.bm,
  Y,
  family = "mgaussian",
  screen = "SSR",
  lambda = lambda_big,
  eps = eps,
  max.iter = 1e5
)
fit_ada <- biglasso(
  X.bm,
  Y,
  family = "mgaussian",
  screen = "Adaptive",
  lambda = lambda_big,
  eps = eps,
  max.iter = 1e5
)

expect_equal(fit_glmnet$lambda, fit_ssr$lambda * sqrt(m))
expect_equal(fit_glmnet$lambda, fit_ada$lambda * sqrt(m))

for (k in 1:m) {
  b_glmnet <- unname(rbind(fit_glmnet$a0[k, ], as.matrix(fit_glmnet$beta[[k]])))
  b_ssr <- unname(as.matrix(fit_ssr$beta[[k]]))
  b_ada <- unname(as.matrix(fit_ada$beta[[k]]))
  expect_equal(b_glmnet, b_ssr, tolerance = tolerance)
  expect_equal(b_glmnet, b_ada, tolerance = tolerance)
}

# beta should be a named list, one sparse matrix (p+1 x nlambda) per response
expect_equal(length(fit_ssr$beta), m)
expect_true(all(sapply(fit_ssr$beta, function(b) all(dim(b) == c(p + 1, length(fit_ssr$lambda))))))

# loss (RSS, summed/averaged appropriately) should be non-increasing as
# lambda decreases (i.e. as the model gets less constrained)
expect_true(all(diff(fit_ssr$loss) <= 1e-8))


# Test parallel computing ---------------------------------------------------

fit_ssr2 <- biglasso(
  X.bm,
  Y,
  family = "mgaussian",
  screen = "SSR",
  lambda = lambda_big,
  eps = eps,
  ncores = 2,
  max.iter = 1e5
)
tol <- 1e-2
for (k in 1:m) {
  expect_equivalent(as.matrix(fit_ssr$beta[[k]]), as.matrix(fit_ssr2$beta[[k]]), tolerance = tol)
}


# Test elastic net / ridge --------------------------------------------------
# (Not compared numerically against glmnet: the two packages' ridge terms
# for the multi-response group penalty are not related by the simple
# sqrt(m) rescaling that holds for pure lasso, so these are sanity/shape
# checks instead of exact-agreement checks.)

fit_enet <- biglasso(
  X.bm,
  Y,
  family = "mgaussian",
  penalty = "enet",
  alpha = 0.5,
  eps = 1e-10,
  max.iter = 1e5
)
expect_equal(fit_enet$screen, "SSR") # Adaptive isn't supported for enet
expect_true(all(sapply(fit_enet$beta, function(b) all(is.finite(as.matrix(b))))))

# ridge should never produce exact zeros -- but note penalty = "ridge" is
# implemented internally as elastic net with alpha = 1e-6, not literal L2-only
# ridge, and lambda_max is computed as zmax / alpha. That inflates ridge's own
# auto-generated lambda path by a factor of ~1/alpha = 1e6, so even the
# *smallest* lambda in that path (default lambda.min = 0.001 * lambda_max) is
# still a very strong penalty in absolute terms -- not the weakly regularized
# regime the density check needs. Evaluating at a lambda drawn from the
# lasso path's scale instead gives genuinely weak regularization (see the
# analogous, empirically-confirmed fix in test_biglasso_cox.r).
fit_ridge <- biglasso(
  X.bm,
  Y,
  family = "mgaussian",
  penalty = "ridge",
  lambda = min(lambda_big),
  eps = 1e-10,
  max.iter = 1e5
)
frac_nonzero <- sapply(fit_ridge$beta, function(b) mean(as.matrix(b[-1, ]) != 0))
expect_true(all(frac_nonzero == 1))


# Test penalty.factor --------------------------------------------------------

pf <- rep(1, p)
pf[1] <- 0 # unpenalized: should remain in the model at every lambda
fit_pf <- biglasso(
  X.bm,
  Y,
  family = "mgaussian",
  screen = "SSR",
  penalty.factor = pf,
  eps = eps,
  max.iter = 1e5
)
expect_true(all(sapply(fit_pf$beta, function(b) all(as.matrix(b[2, ]) != 0))))


# Test dfmax ------------------------------------------------------------------
# dfmax stops the path once the number of nonzero variables exceeds the
# bound; the lambda value that first triggers the stop is itself retained
# (matching glmnet/ncvreg convention), so it's the *only* point allowed to
# exceed dfmax.

fit_dfmax <- biglasso(X.bm, Y, family = "mgaussian", dfmax = 3, eps = eps, max.iter = 1e5)
nv <- sapply(1:ncol(fit_dfmax$beta[[1]]), function(l) {
  max(sapply(fit_dfmax$beta, function(b) sum(b[-1, l] != 0)))
})
expect_true(length(fit_dfmax$lambda) < 100) # path should stop early
expect_true(all(nv[-length(nv)] <= 3)) # dfmax respected until the stopping point
expect_true(nv[length(nv)] > 3) # last retained point is the one that triggered the stop


# Test response names propagate to beta list names --------------------------

Ynamed <- Y
colnames(Ynamed) <- c("resp_A", "resp_B", "resp_C")
fit_named <- biglasso(X.bm, Ynamed, family = "mgaussian", nlambda = 10)
expect_equal(names(fit_named$beta), c("resp_A", "resp_B", "resp_C"))


# Test predict.mbiglasso / coef.mbiglasso ------------------------------------
# (regression test: predict.mbiglasso used to error for type = "link"/
# "response" because of a broken class check, and coef.mbiglasso used to
# silently corrupt its output via append() on a matrix; see git history.)

li <- 20
for (k in 1:m) {
  b <- coef(fit_ssr)[[k]][, li]
  manual_link <- X %*% b[-1] + b[1]

  link <- predict(fit_ssr, X.bm, type = "link", k = k)
  expect_equal(as.numeric(link[, li]), as.numeric(manual_link), tolerance = 1e-6)

  resp <- predict(fit_ssr, X.bm, type = "response", k = k)
  expect_equal(resp, link) # mgaussian has no link function

  coefs <- predict(fit_ssr, X.bm, type = "coefficients", k = k)
  expect_equal(as.matrix(coefs), as.matrix(coef(fit_ssr)[[k]]))
}

nv <- predict(fit_ssr, X.bm, type = "nvars", k = 1)
expect_equal(length(nv), length(fit_ssr$lambda))
expect_true(all(nv == Matrix::colSums(fit_ssr$beta[[1]][-1, , drop = FALSE] != 0)))

vv <- predict(fit_ssr, X.bm, type = "vars", k = 1, which = li)
expect_true(all(vv %in% seq_len(p)))


# Test plot.mbiglasso ---------------------------------------------------------
# smoke test only: plot.mbiglasso() depends on the same coef.mbiglasso() path
# exercised above, and previously errored for any actual mgaussian fit.

tmp_pdf <- tempfile(fileext = ".pdf")
pdf(tmp_pdf)
plot(fit_ssr)
plot(fit_ssr, norm.beta = FALSE)
dev.off()
unlink(tmp_pdf)
expect_true(TRUE) # reaching this point means plot.mbiglasso() didn't error


# Test cv.biglasso() rejects family = "mgaussian" -----------------------------

expect_error(cv.biglasso(X.bm, Y, family = "mgaussian"))
