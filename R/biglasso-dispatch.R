# Dispatch table for biglasso()'s (family, screen) -> C routine mapping.
#
# Each leaf lists:
#   - routine: the .Call() routine name (see src/init.c for registration)
#   - args:    the exact, ordered argument tokens the routine expects. Token
#              values are supplied by biglasso() via a single named list
#              (built once per call, see `dispatch_values` in biglasso())
#              covering every token used anywhere in this table; tokens that
#              don't apply to a given family (e.g. 'd'/'d_idx' for
#              family != "cox") are simply never selected out of that list.
#   - out:     the names to assign, in order, to the routine's returned
#              list. This is what `res[[k]]` used to spell out at each call
#              site; it's now colocated with the routine name so the two can
#              be audited together instead of hunting for a matching
#              `res[[k]]` block elsewhere in biglasso.R.
#
# A family entry named ".default" (currently only binomial and cox have one)
# is used when `screen` doesn't match any of that family's named entries --
# this reproduces two previously-implicit quirks:
#   - binomial silently falls back to its SSR routine for *any* unrecognized
#     screen value (its original dispatch was an if/Hybrid/else-if/Adaptive/
#     else chain, with SSR as the unconditional else)
#   - cox falls back to the plain (unscreened) cdfit_cox() routine for
#     screen = "Hybrid" or "None" (neither has a dedicated cox routine; its
#     original dispatch was the analogous if/else-if chain with cdfit_cox()
#     as the else)
# families without a ".default" (gaussian, mgaussian) instead error via
# biglasso()'s `if (is.null(spec)) stop("Invalid screening method!")`,
# matching their original switch()'s explicit stop() default arm.
#
# alg.logistic == "MM" is keyed as the pseudo-screen "MM" (binomial only):
# it dispatches to a different routine than any actual screen value, and
# biglasso() selects this key directly rather than via `screen`.

biglasso_dispatch_table <- list(
  gaussian = list(
    Adaptive = list(
      routine = "cdfit_gaussian_ada_edpp_ssr",
      args = c("X", "yy", "row_idx", "lambda", "nlambda", "lambda_log_scale", "lambda_min",
               "alpha", "user_lambda", "eps", "max_iter", "penalty_factor", "dfmax", "ncores",
               "update_thresh", "verbose"),
      out = c("b", "center", "scale", "lambda", "loss", "iter", "rejections",
              "safe_rejections", "col.idx")
    ),
    SSR = list(
      routine = "cdfit_gaussian_ssr",
      args = c("X", "yy", "row_idx", "lambda", "nlambda", "lambda_log_scale", "lambda_min",
               "alpha", "user_lambda", "eps", "max_iter", "penalty_factor", "dfmax", "ncores",
               "verbose"),
      out = c("b", "center", "scale", "lambda", "loss", "iter", "rejections", "col.idx")
    ),
    Hybrid = list(
      routine = "cdfit_gaussian_bedpp_ssr",
      args = c("X", "yy", "row_idx", "lambda", "nlambda", "lambda_log_scale", "lambda_min",
               "alpha", "user_lambda", "eps", "max_iter", "penalty_factor", "dfmax", "ncores",
               "safe_thresh", "verbose"),
      out = c("b", "center", "scale", "lambda", "loss", "iter", "rejections",
              "safe_rejections", "col.idx")
    )
  ),
  binomial = list(
    MM = list(
      routine = "cdfit_binomial_ssr_approx",
      args = c("X", "yy", "row_idx", "lambda", "nlambda", "lambda_min", "alpha", "user_lambda",
               "eps", "max_iter", "penalty_factor", "dfmax", "ncores", "warn", "verbose"),
      out = c("a", "b", "center", "scale", "lambda", "loss", "iter", "rejections", "col.idx")
    ),
    Hybrid = list(
      routine = "cdfit_binomial_slores_ssr",
      args = c("X", "yy", "n_pos", "ylab", "row_idx", "lambda", "nlambda", "lambda_log_scale",
               "lambda_min", "alpha", "user_lambda", "eps", "max_iter", "penalty_factor",
               "dfmax", "ncores", "warn", "safe_thresh", "verbose"),
      out = c("a", "b", "center", "scale", "lambda", "loss", "iter", "rejections",
              "safe_rejections", "col.idx")
    ),
    Adaptive = list(
      routine = "cdfit_binomial_ada_slores_ssr",
      args = c("X", "yy", "n_pos", "ylab", "row_idx", "lambda", "nlambda", "lambda_log_scale",
               "lambda_min", "alpha", "user_lambda", "eps", "max_iter", "penalty_factor",
               "dfmax", "ncores", "warn", "safe_thresh", "update_thresh", "verbose"),
      out = c("a", "b", "center", "scale", "lambda", "loss", "iter", "rejections",
              "safe_rejections", "col.idx")
    ),
    SSR = list(
      routine = "cdfit_binomial_ssr",
      args = c("X", "yy", "row_idx", "lambda", "nlambda", "lambda_log_scale", "lambda_min",
               "alpha", "user_lambda", "eps", "max_iter", "penalty_factor", "dfmax", "ncores",
               "warn", "verbose"),
      out = c("a", "b", "center", "scale", "lambda", "loss", "iter", "rejections", "col.idx")
    ),
    .default = "SSR"  # any unrecognized screen silently behaves like "SSR"
  ),
  cox = list(
    SSR = list(
      routine = "cdfit_cox_ssr",
      args = c("X", "yy", "d", "d_idx", "row_idx_cox", "lambda", "nlambda", "lambda_log_scale",
               "lambda_min", "alpha", "user_lambda", "eps", "max_iter", "penalty_factor",
               "dfmax", "ncores", "warn", "verbose"),
      out = c("b", "center", "scale", "lambda", "loss", "iter", "rejections", "col.idx")
    ),
    sscox = list(
      routine = "cdfit_cox_sscox",
      args = c("X", "yy", "d", "d_idx", "row_idx_cox", "lambda", "nlambda", "lambda_log_scale",
               "lambda_min", "alpha", "user_lambda", "eps", "max_iter", "penalty_factor",
               "dfmax", "ncores", "warn", "safe_thresh", "verbose"),
      out = c("b", "center", "scale", "lambda", "loss", "iter", "rejections", "col.idx")
    ),
    scox = list(
      routine = "cdfit_cox_scox",
      args = c("X", "yy", "d", "d_idx", "row_idx_cox", "lambda", "nlambda", "lambda_log_scale",
               "lambda_min", "alpha", "user_lambda", "eps", "max_iter", "penalty_factor",
               "dfmax", "ncores", "warn", "safe_thresh", "verbose"),
      out = c("b", "center", "scale", "lambda", "loss", "iter", "rejections", "col.idx")
    ),
    safe = list(
      routine = "cdfit_cox_safe",
      args = c("X", "yy", "d", "d_idx", "row_idx_cox", "lambda", "nlambda", "lambda_log_scale",
               "lambda_min", "alpha", "user_lambda", "eps", "max_iter", "penalty_factor",
               "dfmax", "ncores", "warn", "safe_thresh", "verbose"),
      out = c("b", "center", "scale", "lambda", "loss", "iter", "rejections", "col.idx")
    ),
    Adaptive = list(
      routine = "cdfit_cox_ada_scox",
      args = c("X", "yy", "d", "d_idx", "row_idx_cox", "lambda", "nlambda", "lambda_log_scale",
               "lambda_min", "alpha", "user_lambda", "eps", "max_iter", "penalty_factor",
               "dfmax", "ncores", "warn", "safe_thresh", "update_thresh", "verbose"),
      out = c("b", "center", "scale", "lambda", "loss", "iter", "rejections",
              "safe_rejections", "col.idx")
    ),
    None = list(
      routine = "cdfit_cox",
      args = c("X", "yy", "d", "d_idx", "row_idx_cox", "lambda", "nlambda", "lambda_log_scale",
               "lambda_min", "alpha", "user_lambda", "eps", "max_iter", "penalty_factor",
               "dfmax", "ncores", "warn", "verbose"),
      out = c("b", "center", "scale", "lambda", "loss", "iter", "rejections", "col.idx")
    ),
    .default = "None"  # any unrecognized screen (e.g. "Hybrid") behaves like "None"
  ),
  mgaussian = list(
    SSR = list(
      routine = "cdfit_mgaussian_ssr",
      args = c("X", "yy", "row_idx", "lambda", "nlambda", "lambda_log_scale", "lambda_min",
               "alpha", "user_lambda", "eps", "max_iter", "penalty_factor", "dfmax", "ncores",
               "verbose"),
      out = c("b", "center", "scale", "lambda", "loss", "iter", "rejections", "col.idx")
    ),
    Adaptive = list(
      routine = "cdfit_mgaussian_ada",
      args = c("X", "yy", "row_idx", "lambda", "nlambda", "lambda_log_scale", "lambda_min",
               "alpha", "user_lambda", "eps", "max_iter", "penalty_factor", "dfmax", "ncores",
               "safe_thresh", "update_thresh", "verbose"),
      out = c("b", "center", "scale", "lambda", "loss", "iter", "rejections",
              "safe_rejections", "col.idx")
    )
  )
)

# Looks up the dispatch spec for (family, screen), applying MM/.default
# fallbacks. `screen_key` should already be "MM" for alg.logistic == "MM".
biglasso_dispatch_lookup <- function(family, screen_key) {
  family_table <- biglasso_dispatch_table[[family]]
  spec <- family_table[[screen_key]]
  if (is.null(spec) && !is.null(family_table[[".default"]])) {
    spec <- family_table[[family_table[[".default"]]]]
  }
  if (is.null(spec)) stop("Invalid screening method!")
  spec
}
