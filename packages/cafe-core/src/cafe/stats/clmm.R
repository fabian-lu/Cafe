#!/usr/bin/env Rscript
# Fit a cumulative link mixed model (ordinal regression) and print the result as JSON.
#
# Usage:  Rscript clmm.R <csv_path> <comma,separated,factor,columns> [order] [lr_tests: 1|0]
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
lr_tests <- length(args) >= 4 && identical(args[[4]], "1")

fixed_terms <- function(ord) {
  terms <- paste(sprintf("`%s`", factors), collapse = " + ")
  if (ord >= 2 && length(factors) >= 2) sprintf("(%s)^%d", terms, ord) else terms
}

# Type-II likelihood-ratio test per model term T: the model with every term that doesn't
# contain T (T itself included) vs. the same model without T. One test per factor or
# interaction — "does it matter at all?" — where the coefficients only compare a level with
# the baseline. `full` is the already fitted model with all terms (reused, not refit).
lr_term_tests <- function(d, ord, full) {
  tt <- stats::terms(stats::as.formula(paste("~", fixed_terms(ord))))
  labels <- attr(tt, "term.labels")
  inc <- attr(tt, "factors")                       # rows: variables, cols: terms
  sets <- lapply(seq_along(labels), function(j) rownames(inc)[inc[, j] > 0])
  cache <- list()
  fit_terms <- function(idx) {                     # idx: indices into `labels`
    if (length(idx) == length(labels)) return(full)
    key <- paste0("k", paste(sort(idx), collapse = ","))   # never "": R can't look that name up
    if (is.null(cache[[key]])) {
      rhs <- if (length(idx)) paste(labels[idx], collapse = " + ") else "1"
      f <- stats::as.formula(paste("verdict ~ (1|input_id) +", rhs))
      fm <- tryCatch(suppressWarnings(ordinal::clmm(f, data = d, Hess = FALSE)),
                     error = function(e) NULL)
      cache[[key]] <<- list(fit = fm)
    }
    cache[[key]]$fit
  }
  ok <- function(fm) !is.null(fm) && isTRUE(fm$optRes$convergence == 0)
  lapply(seq_along(labels), function(j) {
    bigger <- which(vapply(sets, function(s) length(s) > length(sets[[j]]) &&
                             all(sets[[j]] %in% s), logical(1)))
    with_t <- setdiff(seq_along(labels), bigger)
    mp <- fit_terms(with_t)
    mm <- fit_terms(setdiff(with_t, j))
    out <- list(term = gsub("`", "", labels[j]), factors = I(gsub("`", "", sets[[j]])))
    if (!ok(mp) || !ok(mm)) return(c(out, list(df = NA, chisq = NA, p = NA, converged = FALSE)))
    lp <- logLik(mp); lm0 <- logLik(mm)
    df <- attr(lp, "df") - attr(lm0, "df")
    # no parameters added: the term is confounded with the others (aliased), not testable
    if (df <= 0) return(c(out, list(df = df, chisq = NA, p = NA, converged = TRUE,
                                     estimable = FALSE)))
    chisq <- max(0, 2 * (as.numeric(lp) - as.numeric(lm0)))
    c(out, list(df = df, chisq = chisq, p = stats::pchisq(chisq, df, lower.tail = FALSE),
                converged = TRUE, estimable = TRUE))
  })
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
  # clmm never checks its optimizer's status. nlminb's `convergence` is 0 only on success
  # (?nlminb); after e.g. "false convergence (8)" the Hessian is meaningless and the
  # standard errors can be ~100x too small. Added to the result after the tryCatch.
  optimizer <- list(convergence = m$optRes$convergence, message = m$optRes$message)

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
  vc <- ordinal::VarCorr(m)$input_id   # 1x1 random-intercept variance (+ "stddev" attribute)
  random_intercept <- list(variance = vc[1, 1], sd = unname(attr(vc, "stddev"))[1])

  # per-term likelihood-ratio tests — only for a converged fit (its logLik is the reference)
  factor_tests <- if (lr_tests && isTRUE(optimizer$convergence == 0))
    lr_term_tests(d, used_order, m) else NULL

  # deparse() splits long formulas into several strings (a JSON array); join them into one
  formula <- paste(trimws(deparse(form, width.cutoff = 500L)), collapse = " ")
  list(available = TRUE, n_obs = nrow(d), formula = formula,
       logLik = as.numeric(logLik(m)), coefficients = coeffs, thresholds = thresholds,
       used_order = used_order,   # < the requested order when interactions fell back
       random_intercept = random_intercept, factor_tests = factor_tests)
}, error = function(e) list(available = FALSE, error = conditionMessage(e)))
if (isTRUE(result$available)) result$optimizer <- optimizer

emit(result)
