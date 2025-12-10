# Internal biglasso functions

Internal biglasso functions

## Usage

``` r
loss.biglasso(y, yhat, family, eval.metric, grouped = TRUE)
```

## Arguments

- y:

  The observed response vector.

- yhat:

  The predicted response vector.

- family:

  Either "gaussian" or "binomial", depending on the response.

- eval.metric:

  The evaluation metric for the cross-validated error and for choosing
  optimal `lambda`. "default" for linear regression is MSE (mean squared
  error), for logistic regression is misclassification error. "MAPE",
  for linear regression only, is the Mean Absolute Percentage Error.
  "auc", for logistic regression, is the area under the receiver
  operating characteristic curve (ROC).

- grouped:

  Whether to calculate loss for the entire CV fold (`TRUE`), or for
  predictions individually. Must be `TRUE` when `eval.metric` is 'auc'.

## Details

These are not intended for use by users. `loss.biglasso` calculates the
value of the loss function for the given predictions (used for
cross-validation).

## Author

Yaohui Zeng and Patrick Breheny
