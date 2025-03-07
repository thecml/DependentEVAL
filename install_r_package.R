if (!requireNamespace("compound.Cox", quietly = TRUE)) {
  install.packages("compound.Cox")
}

library("compound.Cox")

ns <- loadNamespace("compound.Cox")

ls(ns)

print(deparse(get("CG.Clayton", envir = ns)))