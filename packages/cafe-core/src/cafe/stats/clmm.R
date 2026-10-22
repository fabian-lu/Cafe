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

fixed_terms <- function(ord) {
  terms <- paste(sprintf("`%s`", factors), collapse = " + ")
  if (ord >= 2 && length(factors) >= 2) sprintf("(%s)^%d", terms, ord) else terms
}

build_formula <- function(ord) {
  stats::as.formula(sprintf("verdict ~ (1|input_id) + %s", fixed_terms(ord)))  # random effect first
}

# The factor(s) and level(s) behind every fixed-effect coefficient, keyed by the name R
# gives that coefficient (e.g. `modelllama3:8b:promptcot`). The caller uses this instead of
# splitting that name on ':', which can't tell an interaction from a level that contains
# ':' (ollama tags such as llama3:8b), and which misses back-quoted non-syntactic factor
# names (`top-k`5). Covers every order up to `ord`, so it also fits the main-effects fallback.
coef_info <- function(d, ord) {
  inc <- attr(stats::terms(stats::as.formula(paste("~", fixed_terms(ord)))), "factors")
  if (nrow(inc) != length(factors)) return(list())
  info <- list()
  for (t in seq_len(ncol(inc))) {
    idx <- which(inc[, t] > 0)   # rows follow the order of `factors`
    # treatment contrasts: one column per non-baseline level combination
    lv <- expand.grid(lapply(factors[idx], function(f) levels(d[[f]])[-1]), stringsAsFactors = FALSE)
    for (i in seq_len(nrow(lv))) {
      lvl <- unlist(lv[i, ], use.names = FALSE)
      name <- paste0(rownames(inc)[idx], lvl, collapse = ":")   # R's own label for the column
      info[[name]] <- list(factors = I(factors[idx]), levels = I(lvl))
    }
  }
  info
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
  used_order <- order
  if (is.null(m) && order >= 2) {   # interactions not estimable -> fall back to main effects
    form <- build_formula(1L)
    m <- ordinal::clmm(form, data = d, Hess = TRUE)
    used_order <- 1L
  }
  if (is.null(m)) stop("model did not converge")

  ct <- coef(summary(m))
  beta_names <- names(m$beta)          # fixed-effect (factor) coefficients
  alpha_names <- names(m$alpha)        # threshold (cut-point) coefficients

  info <- coef_info(d, order)

  coeffs <- lapply(beta_names, function(rn) {
    row <- ct[rn, ]
    c(list(term = rn, estimate = unname(row[[1]]), std_error = unname(row[[2]]),
           z = unname(row[[3]]), p = unname(row[[4]])), info[[rn]])
  })
  thresholds <- lapply(alpha_names, function(rn) {
    list(term = rn, estimate = unname(ct[rn, 1]))
  })

  list(available = TRUE, n_obs = nrow(d), formula = deparse(form),
       logLik = as.numeric(logLik(m)), coefficients = coeffs, thresholds = thresholds,
       used_order = used_order)   # < the requested order when interactions fell back
}, error = function(e) list(available = FALSE, error = conditionMessage(e)))

emit(result)
