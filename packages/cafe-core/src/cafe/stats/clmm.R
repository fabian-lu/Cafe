#!/usr/bin/env Rscript
# Fit a cumulative link mixed model (ordinal regression) and print the result as JSON.
#
# Usage:  Rscript clmm.R <csv_path> <comma,separated,factor,columns>
# The CSV must contain: `verdict` (integer ordinal score), `input_id` (the grouping
# variable for the random intercept), and one column per factor.
#
# Output (stdout): a JSON object. On any failure it still prints valid JSON with
# available=false and an error message, so the Python caller never has to parse
# tracebacks.

suppressWarnings(suppressMessages({
  have_ordinal <- requireNamespace("ordinal", quietly = TRUE)
  have_json <- requireNamespace("jsonlite", quietly = TRUE)
}))

emit <- function(obj) {
  if (have_json) {
    cat(jsonlite::toJSON(obj, auto_unbox = TRUE, na = "null", digits = 8))
  } else {
    cat('{"available": false, "error": "R package jsonlite not installed (install.packages(\\"jsonlite\\"))"}')
  }
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  emit(list(available = FALSE, error = "usage: clmm.R <csv> <factors>"))
  quit(status = 0)
}
csv_path <- args[[1]]
factors <- strsplit(args[[2]], ",")[[1]]
order <- if (length(args) >= 3) suppressWarnings(as.integer(args[[3]])) else 1L
if (is.na(order)) order <- 1L

build_formula <- function(ord) {
  terms <- paste(sprintf("`%s`", factors), collapse = " + ")
  fixed <- if (ord >= 2 && length(factors) >= 2) sprintf("(%s)^%d", terms, ord) else terms
  stats::as.formula(sprintf("verdict ~ (1|input_id) + %s", fixed))  # random effect first
}

if (!have_ordinal) {
  emit(list(available = FALSE,
            error = "R package 'ordinal' not installed (run: Rscript -e 'install.packages(\"ordinal\")')"))
  quit(status = 0)
}

result <- tryCatch({
  d <- read.csv(csv_path, stringsAsFactors = TRUE, check.names = FALSE)
  d$verdict <- ordered(d$verdict)
  d$input_id <- factor(d$input_id)
  # Treat every declared factor as categorical (so a numeric-valued knob like top_k=[1,2]
  # is levels, not a continuous covariate — consistent with the Gaussian model).
  for (f in factors) d[[f]] <- factor(d[[f]])

  form <- build_formula(order)
  m <- tryCatch(ordinal::clmm(form, data = d, Hess = TRUE), error = function(e) NULL)
  if (is.null(m) && order >= 2) {   # interactions not estimable -> fall back to main effects
    form <- build_formula(1L)
    m <- ordinal::clmm(form, data = d, Hess = TRUE)
  }
  if (is.null(m)) stop("model did not converge")

  ct <- coef(summary(m))
  beta_names <- names(m$beta)          # fixed-effect (factor) coefficients
  alpha_names <- names(m$alpha)        # threshold (cut-point) coefficients

  coeffs <- lapply(beta_names, function(rn) {
    row <- ct[rn, ]
    list(term = rn, estimate = unname(row[[1]]), std_error = unname(row[[2]]),
         z = unname(row[[3]]), p = unname(row[[4]]))
  })
  thresholds <- lapply(alpha_names, function(rn) {
    list(term = rn, estimate = unname(ct[rn, 1]))
  })

  list(available = TRUE, n_obs = nrow(d), formula = deparse(form),
       logLik = as.numeric(logLik(m)), coefficients = coeffs, thresholds = thresholds)
}, error = function(e) list(available = FALSE, error = conditionMessage(e)))

emit(result)
