.onUnload <- function (libpath) {
  library.dynam.unload("biglasso", libpath)
}
