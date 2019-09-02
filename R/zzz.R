.onUnload <- function (libpath) {
  library.dynam.unload("biglasso", libpath)
  try(
    system("rm -R /tmp/boost_interprocess",
           intern = TRUE,
           ignore.stderr = TRUE)
    )
}
