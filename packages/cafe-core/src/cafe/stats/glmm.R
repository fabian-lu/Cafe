#!/usr/bin/env Rscript
# Fit a logistic mixed model (binary GLMM) and print the result as JSON.
#
# Usage:  Rscript glmm.R <csv_path> <comma,separated,factor,columns> [interaction_order]
# The CSV must contain: `verdict` (0/1 pass/fail), `input_id` (the grouping variable for
# the random intercept), and one column per factor.
#
# Model:  verdict ~ (1 | input_id) + <factors>, family = binomial(logit)
# This is the binary analogue of the ordinal CLMM (clmm.R) and the numeric linear mixed
# model — a per-question random intercept, so factor effects are estimated net of
# question-to-question difficulty (conditional / subject-specific odds ratios).
#
# Output (stdout): a JSON object. On any failure it still prints valid JSON with
# available=false and an error message, so the Python caller never parses tracebacks.

suppressWarnings(suppressMessages({
  have_lme4 <- requireNamespace("lme4", quietly = TRUE)
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
  emit(list(available = FALSE, error = "usage: glmm.R <csv> <factors>"))
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

if (!have_lme4) {
  emit(list(available = FALSE,
            error = "R package 'lme4' not installed (run: Rscript -e 'install.packages(\"lme4\")')"))
  quit(status = 0)
}

result <- tryCatch({
  d <- read.csv(csv_path, stringsAsFactors = TRUE, check.names = FALSE)
  d$verdict <- as.numeric(as.character(d$verdict))   # 0/1 for the binomial family
  d$input_id <- factor(d$input_id)
  # Treat every declared factor as categorical (a numeric-valued knob like top_k=[1,2] is
  # levels, not a continuous covariate — consistent with the CLMM / Gaussian models).
  for (f in factors) d[[f]] <- factor(d[[f]])

  fit_one <- function(ord) {
    suppressWarnings(suppressMessages(
      lme4::glmer(build_formula(ord), data = d, family = stats::binomial())
    ))
  }
  m <- tryCatch(fit_one(order), error = function(e) NULL)
  used_order <- order
  if (is.null(m) && order >= 2) {   # interactions not estimable -> fall back to main effects
    m <- tryCatch(fit_one(1L), error = function(e) NULL)
    used_order <- 1L
  }
  if (is.null(m)) stop("model did not converge")

  ct <- as.data.frame(summary(m)$coefficients)   # Estimate, Std. Error, z value, Pr(>|z|)
  rn <- rownames(ct)
  keep <- rn != "(Intercept)"
  coeffs <- lapply(which(keep), function(i) {
    list(term = rn[i], estimate = ct[i, 1], std_error = ct[i, 2],
         z = ct[i, 3], p = ct[i, 4])
  })

  list(available = TRUE, n_obs = nrow(d), formula = deparse(build_formula(used_order)),
       logLik = as.numeric(logLik(m)), coefficients = coeffs)
}, error = function(e) list(available = FALSE, error = conditionMessage(e)))

emit(result)
