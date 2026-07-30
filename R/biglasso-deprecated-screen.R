# Deprecated `screen` method names, mapped to which warning group they
# belong to. Each group (see biglasso_deprecated_screen_groups below) has
# its own wording and its own current-equivalent replacement; grouping
# lets that live in one place per group instead of being re-typed for
# every old name that maps to it.
biglasso_deprecated_screen_map <- c(
  "SSR-BEDPP" = "hybrid",
  "SSR-Slores" = "hybrid",
  "SSR-Slores-Batch" = "adaptive",
  "SEDPP-Batch" = "adaptive",
  "SEDPP-Batch-SSR" = "adaptive",
  "SEDPP-Batchfix-SSR" = "adaptive",
  "SEDPP" = "removed",
  "SSR-Dome" = "removed",
  "NS-NAC" = "removed",
  "SSR-NAC" = "removed",
  "SEDPP-NAC" = "removed",
  "SSR-Dome-NAC" = "removed",
  "SSR-BEDPP-NAC" = "removed",
  "SSR-Slores-NAC" = "removed"
)

biglasso_deprecated_screen_groups <- list(
  hybrid = list(
    new = "Hybrid",
    message = "Hybrid screen methods (\"SSR-BEDPP\", \"SSR-Slores\") will be renamed as \"Hybrid\". Automatically switching to \"Hybrid\" screen method"
  ),
  adaptive = list(
    new = "Adaptive",
    message = "Adaptive or batch screen methods (\"SSR-Slores-Batch\", \"SEDPP-Batch\", \"SEDPP-Batch-SSR\", \"SEDPP-Batchfix-SSR\") will be renamed as \"Adaptive\". Automatically switching to \"Adaptive\" screen method."
  ),
  removed = list(
    new = "Adaptive",
    message = "The following screen methods will be removed:\n\"SEDPP\", \"SSR-Dome\", \"NS-NAC\", \"SSR-NAC\", \"SEDPP-NAC\",\"SSR-Dome-NAC\", \"SSR-BEDPP-NAC\", \"SSR-Slores-NAC\".\nAutomatically switching to \"Adaptive\" screen method."
  )
)
