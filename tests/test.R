if (requireNamespace("tinytest", quietly=TRUE)) {
  if (identical(tolower(Sys.getenv("R_FORCE_TEST")), "true")) {
    tinytest::test_package("biglasso", pattern="^[^_].*\\.[rR]$")
  }
}
