library(testthat)
library(biglasso)

# Get a list of all files in the tests directory
test_files <- list.files("tests", full.names = TRUE)

# Exclude files that begin with an underscore
test_files <- test_files[!grepl("/_[^/]*\\.R$", test_files)]

# Run tests
test_check(test_files)

# test_check("biglasso")