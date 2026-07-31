# Shared plotting body behind plot.biglasso() and plot.mbiglasso(): draws
# the empty plot frame, the coefficient-path lines, and the zero line.
# Callers differ only in how they prepare `Y` (the p x nlambda matrix of
# paths to plot) and `xlab` above this point, and in what the y-axis label
# should say -- `draw_ylab` is called with no arguments to draw it (via
# mtext()), but only when the caller hasn't already supplied their own
# `ylab` via `...`.
plot_biglasso_lines <- function(l, Y, xlab, alpha, draw_ylab, ...) {
  Y <- as.matrix(Y)
  p <- nrow(Y)

  plot.args <- list(
    x = l,
    y = 1:length(l),
    ylim = range(Y),
    xlab = xlab,
    ylab = "",
    type = "n",
    xlim = rev(range(l)),
    las = 1
  )
  new.args <- list(...)
  if (length(new.args)) {
    plot.args[names(new.args)] <- new.args
  }
  do.call("plot", plot.args)
  if (!is.element("ylab", names(new.args))) {
    draw_ylab()
  }

  cols <- hcl(h = seq(15, 375, len = max(4, p + 1)), l = 60, c = 150, alpha = alpha)
  cols <- if (p == 2) cols[c(1, 3)] else cols[1:p]
  line.args <- list(col = cols, lwd = 1 + 2 * exp(-p / 20), lty = 1)
  if (length(new.args)) {
    line.args[names(new.args)] <- new.args
  }
  line.args$x <- l
  line.args$y <- t(Y)
  do.call("matlines", line.args)

  abline(h = 0)
}
