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
  # glmer's own convergence verdict, which suppressWarnings() above hides: the optimizer's
  # code (0 = success, ?lmerControl) and lme4's derivative checks (?convergence; a negative
  # code, e.g. "Model failed to converge with max|grad| = ..."). After such a failure the
  # standard errors can be ~100x too small. Added to the result after the tryCatch.
  conv <- m@optinfo$conv
  optimizer <- list(convergence = as.integer(conv$opt),
                    message = as.character(c(m@optinfo$message, "")[1]),
                    lme4_code = I(as.integer(conv$lme4$code)),
                    lme4_messages = I(as.character(unlist(conv$lme4$messages))))

  info <- coef_info(d, order)
  ct <- as.data.frame(summary(m)$coefficients)   # Estimate, Std. Error, z value, Pr(>|z|)
  rn <- rownames(ct)
  keep <- rn != "(Intercept)"
  coeffs <- lapply(which(keep), function(i) {
    c(list(term = rn[i], estimate = ct[i, 1], std_error = ct[i, 2],
           z = ct[i, 3], p = ct[i, 4]), info[[rn[i]]])
  })
  vc <- as.data.frame(lme4::VarCorr(m))   # grp, var1, var2, vcov (variance), sdcor (sd)
  vc <- vc[vc$grp == "input_id", ]
  random_intercept <- list(variance = vc$vcov[1], sd = vc$sdcor[1])

  # deparse() splits long formulas into several strings (a JSON array); join them into one
  formula <- paste(trimws(deparse(build_formula(used_order), width.cutoff = 500L)), collapse = " ")
  list(available = TRUE, n_obs = nrow(d), formula = formula,
       logLik = as.numeric(logLik(m)), coefficients = coeffs,
       used_order = used_order,   # < the requested order when interactions fell back
       random_intercept = random_intercept)
}, error = function(e) list(available = FALSE, error = conditionMessage(e)))
if (isTRUE(result$available)) result$optimizer <- optimizer

emit(result)
