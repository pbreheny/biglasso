.onUnload <- function (libpath) {
  try(
    system("rm -R /tmp/boost_interprocess",
           intern = TRUE,
           ignore.stderr = TRUE)
    )
  library.dynam.unload("biglasso", libpath)
}
