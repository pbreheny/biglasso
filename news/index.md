# Changelog

## biglasso 1.6.1

CRAN release: 2025-03-05

- Various internal fixes (see below)
- Updating references
- Fixing some broken links
- Removing an OMP directive that was causing stack imbalance issues
- Improved CI testing
- Eliminating use of PROTECT in cpp code
- Some NAMESPACE changes

## biglasso 1.6.0

CRAN release: 2024-04-21

- New: functions biglasso_fit() and biglasso_path(), which allow users
  to turn off standardization and intercept

## biglasso 1.5.2

CRAN release: 2022-10-05

- Update coercion for compatibility with Matrix 1.5
- Now using GitHub Actions instead of Travis for CI

## biglasso 1.5.1

CRAN release: 2022-03-09

- Internal Cpp changes: initialize Xty, remove unused cutoff variable
  ([\#48](https://github.com/pbreheny/biglasso/issues/48))
- Eliminate CV test against ncvreg (the two packages no longer use the
  same approach ([\#47](https://github.com/pbreheny/biglasso/issues/47))

## biglasso 1.5.0

CRAN release: 2022-02-08

- Update headers to maintain compatibility with new version of Rcpp
  ([\#40](https://github.com/pbreheny/biglasso/issues/40))

## biglasso 1.4-1

- changed R package maintainer to Chuyi Wang (<wwaa0208@gmail.com>)
- fixed bugs
- Add ‘auc’, ‘class’ options to cv.biglasso eval.metric
- predict.cv now predicts standard error over CV folds by default; set
  ‘grouped’ argument to FALSE for old behaviour.
- predict.cv.biglasso accepts ‘lambda.min’, ‘lambda.1se’ argument,
  similar to predict.cv.glmnet()

## biglasso 1.4-0

- adaptive screening methods were implemented and set as default when
  applicable
- added sparse Cox regression
- removed uncompetitive screening methods and combined naming of
  screening methods
- version 1.4-0 for CRAN submission

## biglasso 1.3-7

CRAN release: 2019-09-09

- update email to personal email
- coef(cvfit) returns only nonzero cells, as a labelled vector
- set HSR rules as default
- option for non-standardization

## biglasso 1.3-6

CRAN release: 2017-05-04

- optimized the code for computing the slores rule.
- added Slores screening without active cycling (-NAC) for logistic
  regression, research usage only.
- corrected BEDPP for elastic net.
- fixed a bug related to “exporting SSR-BEDPP”.

## biglasso 1.3-5

CRAN release: 2017-04-04

- redocumented using Roxygen2.
- registered native routines for faster and more stable performance.

## biglasso 1.3-4

- fixed a bug related to `dfmax` option. (thanks you Florian Privé!)

## biglasso 1.3-3

CRAN release: 2017-01-26

- fixed bugs related to KKT checking for elastic net. (thanks you
  Florian Privé!)
- added references for screening rules and the technical paper of
  biglasso package.

## biglasso 1.3-2

- added screening methods without active cycling (-NAC) for comparison,
  research usage only.
- fixed a bug related to numeric comparison in Dome test.

## biglasso 1.3-1

CRAN release: 2016-12-31

- fixed bug in SSR-Slores related to numeric equality comparison.

## biglasso 1.3-0

CRAN release: 2016-12-21

- version 1.3-0 for CRAN submission.

## biglasso 1.2-6

- added a newly proposed screening rule, SSR-Slores, for lasso-penalized
  logistic regression.
- added SSR-BEDPP for elastic-net-penalized linear regression.

## biglasso 1.2-5

- updated README.md with benchmarking results.
- added tutorial (vignette).

## biglasso 1.2-4

- added gaussian.cpp: solve lasso without screening, for research only.
- added tests.

## biglasso 1.2-3

CRAN release: 2016-11-14

- changed convergence criteria of logistic regression to be the same as
  that in glmnet.
- optimized source code; preparing for CRAN submission.
- fixed memory leaks occurred on Windows.

## biglasso 1.2-2

- added internal data set: the colon cancer data.

## biglasso 1.2-1

- Implemented another new screening rule (SSR-BEDPP), also combining
  hybrid strong rule with a safe rule (BEDPP).
- implemented EDPP rule with active set cycling strategy for linear
  regression.
- changed convergence criteria to be the same as that in glmnet.

## biglasso 1.1-2

- fixed bugs occurred when some features have identical values for
  different observations. These features are internally removed from
  model fitting.

## biglasso 1.1-1

- Three sparse screening rules (SSR, EDPP, SSR-Dome) were implemented.
  Our new proposed HSR-Dome combines HSR and Dome test for feature
  screening, leading to even better performance as compared to ‘glmnet’.
- OpenMP parallel computing was added to speedup single model fitting.
- Both exact Newton and majorization-minimization (MM) algorithm for
  logistic regression were implemented. The latter could be faster,
  especially in data-larger-than-RAM cases.
- Source code were rewritten in pure cpp.
- Sparse matrix representation was added using Armadillo library.

## biglasso 1.0-1

CRAN release: 2016-03-02

- package ready for CRAN submission.
