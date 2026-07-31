if (interactive()) {
  library(tinytest)
}
library(bigmemory)

# Test setupX() ---------------------------------------------------------------
# setupX() must be called with a bare filename (not a path) plus 'dir', since
# it builds the default backingfile/descriptorfile names by splitting
# 'filename' on ".", which breaks if 'filename' itself contains a path.

set.seed(1)
n <- 50
p <- 5
X <- matrix(round(rnorm(n * p), 4), n, p)

tmp_dir <- tempfile("biglasso_setupX_")
dir.create(tmp_dir)
data_file <- file.path(tmp_dir, "dat.csv")
write.table(X, file = data_file, sep = ",", row.names = FALSE, col.names = FALSE)

old_wd <- setwd(tmp_dir)
X.bm <- setupX("dat.csv", dir = tmp_dir)
setwd(old_wd)

expect_true(inherits(X.bm, "big.matrix"))
expect_equal(dim(X.bm), c(n, p))
expect_equivalent(X.bm[,], X)
expect_true(file.exists(file.path(tmp_dir, "dat.bin")))
expect_true(file.exists(file.path(tmp_dir, "dat.desc")))

# re-attach from the descriptor without re-reading the raw file
X.bm2 <- attach.big.matrix(file.path(tmp_dir, "dat.desc"))
expect_equivalent(X.bm2[,], X)

# custom backing/descriptor file names, tab-separated
data_file2 <- file.path(tmp_dir, "dat2.txt")
write.table(X, file = data_file2, sep = "\t", row.names = FALSE, col.names = FALSE)
setwd(tmp_dir)
X.bm3 <- setupX(
  "dat2.txt",
  dir = tmp_dir,
  sep = "\t",
  backingfile = "custom.bin",
  descriptorfile = "custom.desc"
)
setwd(old_wd)
expect_true(file.exists(file.path(tmp_dir, "custom.bin")))
expect_true(file.exists(file.path(tmp_dir, "custom.desc")))
expect_equivalent(X.bm3[,], X)

# the resulting big.matrix is directly usable as biglasso()'s X
b <- rnorm(p)
y <- rnorm(n, X %*% b)
fit <- biglasso(X.bm, y)
expect_true(inherits(fit, "biglasso"))

unlink(tmp_dir, recursive = TRUE)


# Test plot.biglasso() for gaussian/binomial -----------------------------------
# smoke tests only: family = "cox" is covered in test_biglasso_cox.r, which
# exercises the length(penalty.factor) == nrow(beta) branch (no intercept
# row). Gaussian/binomial fits have an intercept row, so plot.biglasso() must
# instead take the coef(x)[-1, , drop = FALSE] branch -- previously untested.

n <- 50
p <- 10
X <- matrix(rnorm(n * p), n, p)
b <- c(rnorm(3), rep(0, p - 3))
y <- rnorm(n, X %*% b)
X.bm <- as.big.matrix(X)
fit_gauss <- biglasso(X.bm, y, nlambda = 20)

y_bin <- rbinom(n, 1, 0.5)
fit_bin <- biglasso(X.bm, y_bin, family = "binomial", nlambda = 20)

tmp_pdf <- tempfile(fileext = ".pdf")
pdf(tmp_pdf)
plot(fit_gauss)
plot(fit_gauss, log.l = FALSE)
plot(fit_gauss, alpha = 0.5, main = "custom")
plot(fit_bin)
dev.off()
unlink(tmp_pdf)
expect_true(TRUE) # reaching this point means plot.biglasso() didn't error


# Test cv.biglasso() + summary.cv.biglasso() -----------------------------------
# cv.biglasso() itself is already exercised (and checked against cv.glmnet())
# in test_biglasso_logistic.r; here the focus is summary.cv.biglasso(), which
# had no test coverage at all.

cvfit_gauss <- cv.biglasso(X.bm, y, seed = 1, nfolds = 5, ncores = 1, nlambda = 20)
s_gauss <- summary(cvfit_gauss)

expect_true(inherits(s_gauss, "summary.cv.biglasso"))
expect_equal(s_gauss$penalty, "lasso")
expect_equal(s_gauss$model, "linear")
expect_equal(s_gauss$n, n)
expect_equal(s_gauss$p, p)
expect_equal(s_gauss$min, cvfit_gauss$min)
expect_equal(s_gauss$lambda, cvfit_gauss$lambda)
expect_equal(s_gauss$cve, cvfit_gauss$cve)
expect_equal(s_gauss$sigma, sqrt(cvfit_gauss$cve))
rsq_gauss <- pmin(pmax(1 - cvfit_gauss$cve / cvfit_gauss$null.dev, 0), 1)
expect_equal(s_gauss$r.squared, rsq_gauss)
expect_equal(s_gauss$snr, rsq_gauss / (1 - rsq_gauss))
expect_false("pe" %in% names(s_gauss)) # pe is binomial-only; note $pe partial-matches $penalty
expect_equal(s_gauss$nvars, predict(cvfit_gauss$fit, lambda = cvfit_gauss$lambda, type = "nvars"))

out_gauss <- capture.output(print(s_gauss))
expect_true(any(grepl("lasso-penalized linear regression with n=50, p=10", out_gauss)))
expect_true(any(grepl("Scale estimate \\(sigma\\)", out_gauss)))
expect_false(any(grepl("Prediction error", out_gauss)))

out_gauss_1digit <- capture.output(print(s_gauss, digits = 1))
expect_true(any(grepl(
  paste0("R-squared: ", formatC(max(s_gauss$r.squared), 1, format = "f")),
  out_gauss_1digit
)))

cvfit_bin <- cv.biglasso(
  X.bm,
  y_bin,
  family = "binomial",
  seed = 1,
  nfolds = 5,
  ncores = 1,
  nlambda = 20
)
s_bin <- summary(cvfit_bin)

expect_equal(s_bin$model, "logistic")
expect_equal(s_bin$pe, cvfit_bin$pe)
expect_null(s_bin$sigma)
rsq_bin <- pmin(pmax(1 - exp(cvfit_bin$cve - cvfit_bin$null.dev), 0), 1)
expect_equal(s_bin$r.squared, rsq_bin)

out_bin <- capture.output(print(s_bin))
expect_true(any(grepl("lasso-penalized logistic regression with n=50, p=10", out_bin)))
expect_true(any(grepl("Prediction error", out_bin)))
expect_false(any(grepl("Scale estimate", out_bin)))


# Test plot.cv.biglasso() ------------------------------------------------------
# smoke tests, plus checks that the two family-specific error paths (inherited
# from ncvreg::plot.cv.ncvreg -- 'scale' is undefined for binomial, 'pred'
# needs $pe, which only exists for binomial) are reachable through the
# class-reassignment wrapper in plot.cv.biglasso().

pdf(tmp_pdf <- tempfile(fileext = ".pdf"))
plot(cvfit_gauss)
plot(cvfit_gauss, type = "rsq")
plot(cvfit_gauss, type = "snr")
plot(cvfit_gauss, type = "scale")
plot(cvfit_gauss, type = "all")
expect_error(plot(cvfit_gauss, type = "pred"))

plot(cvfit_bin)
plot(cvfit_bin, type = "pred")
plot(cvfit_bin, type = "all")
expect_error(plot(cvfit_bin, type = "scale"))
dev.off()
unlink(tmp_pdf)
expect_true(TRUE) # reaching this point means the non-error plot calls above didn't error
