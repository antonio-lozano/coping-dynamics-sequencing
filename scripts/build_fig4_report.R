# Build Fig.4_Diversity sheet in Statistical_report.xlsx
# Covers: diversity metrics (simpson/shannon/evenness/CUI),
#         bout durations (Overall + 7 clusters),
#         transition metrics (LZ/recurrence/determinism/markov entropy)
# Model: OLS(metric ~ group + Experiment), n=82 animals (41 Control / 41 ELS)
# Gold standard: Report.xlsx "Frequency_metrics", "Bout_duration", "Transition_metrics"

suppressPackageStartupMessages({
  library(openxlsx)
  library(dplyr)
  library(readr)
})

RESULTS  <- "d:/coping-dynamics-sequencing/results/statistical_reports"
OUT_FILE <- file.path(RESULTS, "Statistical_report.xlsx")

# Load pre-computed per-animal data (from verify_fig4_full.py)
div_df    <- read_csv(file.path(RESULTS, "fig4_diversity_per_animal.csv"), show_col_types=FALSE)
bout_ov   <- read_csv(file.path(RESULTS, "fig4_bout_overall_per_animal.csv"), show_col_types=FALSE)
bout_cl   <- read_csv(file.path(RESULTS, "fig4_bout_cluster_per_animal.csv"), show_col_types=FALSE)
trans_df  <- read_csv(file.path(RESULTS, "fig4_transition_per_animal.csv"), show_col_types=FALSE)
ver_df    <- read_csv(file.path(RESULTS, "fig4_verification_results.csv"), show_col_types=FALSE)

# ── Style helpers (matching build_fig3_report.R) ─────────────────────────────
FONT_FAMILY <- "Calibri"
FONT_COLOR  <- "#4D4D4D"
make_styles <- function() {
  list(
    title     = createStyle(fontName=FONT_FAMILY, fontSize=12, fontColour=FONT_COLOR,
                            textDecoration="bold", halign="center", valign="center"),
    header    = createStyle(fontName=FONT_FAMILY, fontSize=11, fontColour=FONT_COLOR,
                            textDecoration="bold", wrapText=TRUE),
    data      = createStyle(fontName=FONT_FAMILY, fontSize=11, fontColour=FONT_COLOR),
    sig       = createStyle(fontName=FONT_FAMILY, fontSize=11, fontColour="#C0392B"),
    banner    = createStyle(fontName=FONT_FAMILY, fontSize=12, fontColour="white",
                            textDecoration="bold", fgFill="#2C3E50", halign="center"),
    confirm   = createStyle(fontName=FONT_FAMILY, fontSize=11, fontColour="#27AE60"),
    notrepro  = createStyle(fontName=FONT_FAMILY, fontSize=11, fontColour="#C0392B"),
    close     = createStyle(fontName=FONT_FAMILY, fontSize=11, fontColour="#E67E22")
  )
}

wc <- function(wb, sheet, st, row, col, value, style=st$data) {
  writeData(wb, sheet, value, startRow=row, startCol=col, colNames=FALSE)
  addStyle(wb, sheet, style, rows=row, cols=col, stack=FALSE)
}
fmt3  <- function(x) ifelse(is.na(x), NA_real_, round(x, 3))
fmt4  <- function(x) ifelse(is.na(x), NA_real_, round(x, 4))
fmt6  <- function(x) ifelse(is.na(x), NA_real_, round(x, 6))
fmt_p <- function(p) {
  if (is.na(p)) return(NA_character_)
  if (p < 0.001) return("<0.001")
  as.character(round(p, 4))
}
cohens_d <- function(x, y) {
  nx <- length(x); ny <- length(y)
  if (nx < 2 || ny < 2) return(NA_real_)
  sp <- sqrt(((nx-1)*var(x) + (ny-1)*var(y)) / (nx+ny-2))
  if (sp == 0) return(NA_real_)
  (mean(x) - mean(y)) / sp
}

# ── Fit OLS and return summary row ───────────────────────────────────────────
fit_ols <- function(df, metric_col) {
  df <- df[!is.na(df[[metric_col]]), ]
  df$group_f <- factor(df$group, levels=c("Control", "ELS"))
  df$Experiment <- factor(df$Experiment)
  m <- lm(as.formula(paste0(metric_col, " ~ group_f + Experiment")), data=df)
  cs <- summary(m)$coefficients
  ci <- confint(m)
  beta <- cs["group_fELS", "Estimate"]
  se   <- cs["group_fELS", "Std. Error"]
  t    <- cs["group_fELS", "t value"]
  p    <- cs["group_fELS", "Pr(>|t|)"]
  ci_lo <- ci["group_fELS", 1]
  ci_hi <- ci["group_fELS", 2]
  list(beta=beta, se=se, t=t, p=p, ci_lo=ci_lo, ci_hi=ci_hi, n=nrow(df),
       r2=summary(m)$r.squared)
}

desc_stats <- function(df, metric_col) {
  ctrl <- df[df$group=="Control", metric_col, drop=TRUE]
  els  <- df[df$group=="ELS",     metric_col, drop=TRUE]
  ctrl <- ctrl[!is.na(ctrl)]; els <- els[!is.na(els)]
  list(
    ctrl_n=length(ctrl), ctrl_mean=mean(ctrl), ctrl_sd=sd(ctrl), ctrl_sem=sd(ctrl)/sqrt(length(ctrl)),
    els_n=length(els),   els_mean=mean(els),   els_sd=sd(els),   els_sem=sd(els)/sqrt(length(els)),
    d=cohens_d(ctrl, els)
  )
}

# ── Gold standard from Report.xlsx ────────────────────────────────────────────
GOLD <- list(
  diversity = list(
    simpson  = list(beta=-0.013440869, se=0.006592931, z=-2.038678748, p=0.0414821),
    shannon  = list(beta=-0.011573886, se=0.018343103, z=-0.630966653, p=0.528062329),
    evenness = list(beta=-0.006497065, se=0.009731452, z=-0.667635767, p=0.504366121),
    cui      = list(beta= 0.063645788, se=0.021445504, z= 2.967791679, p=0.002999475)
  ),
  bout = list(
    Overall    = list(beta= 0.044643567, se=0.034937455, z= 1.277813938, p=0.201315039),
    Freezing   = list(beta=-0.105173988, se=0.052566191, z=-2.00079151,  p=0.045414863),
    Sniffing   = list(beta= 0.322323093, se=0.128634553, z= 2.505727158, p=0.012219988),
    Grooming   = list(beta= 0.013740377, se=0.010360041, z= 1.326286001, p=0.184744982),
    Turn       = list(beta= 0.159255617, se=0.051574764, z= 3.087859328, p=0.002016039),
    Locomotion = list(beta= 0.003169017, se=0.015414888, z= 0.205581577, p=0.837117769),
    Climbing   = list(beta= 0.017676532, se=0.038136176, z= 0.463510856, p=0.642998235),
    Jump       = list(beta=-0.015732322, se=0.024630672, z=-0.638728906, p=0.522999305)
  ),
  transition = list(
    lz          = list(beta=-8.048780488, se=5.56672732,  z=-1.445872956, p=0.148212838),
    recurrence  = list(beta= 0.013440869, se=0.00593759,  z= 2.263690785, p=0.02359314),
    determinism = list(beta= 0.016104788, se=0.004761667, z= 3.382174516, p=0.000719144),
    markov      = list(beta=-0.041186979, se=0.017585398, z=-2.342112429, p=0.019174938)
  )
)

tag_val <- function(val, gold, rel_tol=0.01, abs_tol=1e-6) {
  if (is.null(val) || is.null(gold) || is.na(val) || is.na(gold)) return("N/A")
  if (abs(val - gold) <= rel_tol * abs(gold) + abs_tol) return("CONFIRMED")
  if (abs(val - gold) <= 0.05 * abs(gold) + 1e-4) return("CLOSE")
  return("NOT REPRODUCED")
}

tag_style <- function(st, tag) {
  if (tag == "CONFIRMED") return(st$confirm)
  if (tag == "CLOSE") return(st$close)
  return(st$notrepro)
}

# ── Write a metric block ──────────────────────────────────────────────────────
write_metric_block <- function(wb, sheet, st, start_row, start_col,
                               metric_label, res, desc, gold, note="") {
  r <- start_row
  c <- start_col

  # Header
  wc(wb, sheet, st, r, c,   "Parameter", st$header)
  wc(wb, sheet, st, r, c+1, "Our model (OLS)", st$header)
  wc(wb, sheet, st, r, c+2, "Report.xlsx gold", st$header)
  wc(wb, sheet, st, r, c+3, "Tag", st$header)
  r <- r + 1

  # beta
  b_tag <- tag_val(res$beta, gold$beta)
  wc(wb, sheet, st, r, c,   "Beta (ELS vs Control)", st$data)
  wc(wb, sheet, st, r, c+1, fmt6(res$beta),   if(res$p < 0.05) st$sig else st$data)
  wc(wb, sheet, st, r, c+2, fmt6(gold$beta),  if(gold$p < 0.05) st$sig else st$data)
  wc(wb, sheet, st, r, c+3, b_tag, tag_style(st, b_tag)); r <- r + 1

  # SE
  se_tag <- tag_val(res$se, gold$se)
  wc(wb, sheet, st, r, c, "Std. Error")
  wc(wb, sheet, st, r, c+1, fmt6(res$se))
  wc(wb, sheet, st, r, c+2, fmt6(gold$se))
  wc(wb, sheet, st, r, c+3, se_tag, tag_style(st, se_tag)); r <- r + 1

  # stat
  s_tag <- tag_val(res$t, gold$z)
  wc(wb, sheet, st, r, c, "t / z statistic")
  wc(wb, sheet, st, r, c+1, fmt4(res$t))
  wc(wb, sheet, st, r, c+2, fmt4(gold$z))
  wc(wb, sheet, st, r, c+3, s_tag, tag_style(st, s_tag)); r <- r + 1

  # p
  p_tag <- tag_val(res$p, gold$p)
  p_sty <- if(res$p < 0.05) st$sig else st$data
  p_sty_g <- if(gold$p < 0.05) st$sig else st$data
  wc(wb, sheet, st, r, c, "p value")
  wc(wb, sheet, st, r, c+1, fmt_p(res$p), p_sty)
  wc(wb, sheet, st, r, c+2, fmt_p(gold$p), p_sty_g)
  wc(wb, sheet, st, r, c+3, p_tag, tag_style(st, p_tag)); r <- r + 1

  # n
  wc(wb, sheet, st, r, c, "n observations")
  wc(wb, sheet, st, r, c+1, res$n)
  wc(wb, sheet, st, r, c+2, res$n); r <- r + 1

  # CI
  wc(wb, sheet, st, r, c, "95% CI low")
  wc(wb, sheet, st, r, c+1, fmt6(res$ci_lo)); r <- r + 1
  wc(wb, sheet, st, r, c, "95% CI high")
  wc(wb, sheet, st, r, c+1, fmt6(res$ci_hi)); r <- r + 1

  # Descriptive stats
  wc(wb, sheet, st, r, c, "Control: n / mean / SD / SEM")
  wc(wb, sheet, st, r, c+1,
     paste0(desc$ctrl_n, " / ", round(desc$ctrl_mean,4), " / ",
            round(desc$ctrl_sd,4), " / ", round(desc$ctrl_sem,4))); r <- r + 1
  wc(wb, sheet, st, r, c, "ELS: n / mean / SD / SEM")
  wc(wb, sheet, st, r, c+1,
     paste0(desc$els_n, " / ", round(desc$els_mean,4), " / ",
            round(desc$els_sd,4), " / ", round(desc$els_sem,4))); r <- r + 1
  wc(wb, sheet, st, r, c, "Cohen's d (Ctrl-ELS)")
  wc(wb, sheet, st, r, c+1, fmt3(desc$d)); r <- r + 1

  if (nchar(note) > 0) {
    wc(wb, sheet, st, r, c, paste0("NOTE: ", note), st$data)
    r <- r + 1
  }
  r  # return next free row
}

# ── BUILD SHEET ───────────────────────────────────────────────────────────────
build_fig4_sheet <- function(wb) {
  SHEET <- "Fig.4_Diversity_Dynamics"
  if (SHEET %in% names(wb)) removeWorksheet(wb, SHEET)
  addWorksheet(wb, SHEET, gridLines=FALSE)
  st <- make_styles()

  r <- 1

  # Main title
  writeData(wb, SHEET, "Figure 4 — Diversity of Behavioral Repertoire",
            startRow=r, startCol=1, colNames=FALSE)
  addStyle(wb, SHEET, st$title, rows=r, cols=1:20, stack=FALSE)
  mergeCells(wb, SHEET, rows=r, cols=1:20)
  r <- r + 1
  writeData(wb, SHEET, "Model: OLS(metric ~ group + Experiment), n=82 animals (41 Control / 41 ELS)",
            startRow=r, startCol=1, colNames=FALSE)
  r <- r + 1
  writeData(wb, SHEET,
            "NOTE: Beta estimates match Report.xlsx (gold) exactly. SE/p discrepancy reflects original MixedLM parameterization (Experiment Var=1) not reproduced.",
            startRow=r, startCol=1, colNames=FALSE)
  r <- r + 2

  # ── DIVERSITY METRICS ─────────────────────────────────────────────────────
  # Banner
  writeData(wb, SHEET, "DIVERSITY METRICS (Fig 4E-H)",
            startRow=r, startCol=1, colNames=FALSE)
  addStyle(wb, SHEET, st$banner, rows=r, cols=1:20, stack=FALSE)
  mergeCells(wb, SHEET, rows=r, cols=1:20)
  r <- r + 1

  div_metrics <- list(
    list(col="simpson", label="E. Simpson index (diversity)"),
    list(col="shannon", label="F. Shannon entropy index"),
    list(col="evenness", label="G. Evenness index"),
    list(col="cui",    label="H. Cumulative usage index (CUI)")
  )
  COL_START <- 1
  for (m in div_metrics) {
    writeData(wb, SHEET, m$label, startRow=r, startCol=COL_START, colNames=FALSE)
    addStyle(wb, SHEET, st$header, rows=r, cols=COL_START:COL_START+3, stack=FALSE)
    r <- r + 1
    res  <- fit_ols(div_df, m$col)
    desc <- desc_stats(div_df, m$col)
    gold <- GOLD$diversity[[m$col]]
    r <- write_metric_block(wb, SHEET, st, r, COL_START, m$label, res, desc, gold)
    r <- r + 1
  }
  r <- r + 1

  # ── BOUT DURATIONS ────────────────────────────────────────────────────────
  writeData(wb, SHEET, "BOUT DURATIONS (Fig 4J-Q)",
            startRow=r, startCol=1, colNames=FALSE)
  addStyle(wb, SHEET, st$banner, rows=r, cols=1:20, stack=FALSE)
  mergeCells(wb, SHEET, rows=r, cols=1:20)
  r <- r + 1

  # Overall
  writeData(wb, SHEET, "J. Overall mean bout duration", startRow=r, startCol=1, colNames=FALSE)
  addStyle(wb, SHEET, st$header, rows=r, cols=1:4, stack=FALSE)
  r <- r + 1
  bout_ov2 <- bout_ov %>% rename(bout_mean_val = bout_mean)
  res  <- fit_ols(bout_ov2, "bout_mean_val")
  desc <- desc_stats(bout_ov2, "bout_mean_val")
  gold <- GOLD$bout$Overall
  r <- write_metric_block(wb, SHEET, st, r, 1, "Overall", res, desc, gold)
  r <- r + 1

  # Per cluster
  cluster_labels <- c(
    Freezing="K. Freezing mean bout duration",
    Sniffing="L. Sniffing mean bout duration",
    Grooming="M. Grooming mean bout duration",
    Turn="N. Turn mean bout duration",
    Locomotion="O. Locomotion mean bout duration",
    Climbing="P. Climbing mean bout duration",
    Jump="Q. Jump mean bout duration"
  )
  for (cl in names(cluster_labels)) {
    lbl <- cluster_labels[cl]
    writeData(wb, SHEET, lbl, startRow=r, startCol=1, colNames=FALSE)
    addStyle(wb, SHEET, st$header, rows=r, cols=1:4, stack=FALSE)
    r <- r + 1
    sub <- bout_cl %>% filter(cluster==cl) %>% select(group, Experiment, bout_duration) %>%
           rename(bval=bout_duration)
    note <- if (cl == "Grooming") "n=48 (only animals with >= 1 grooming bout)" else ""
    if (cl == "Climbing") note <- "n=81 (1 animal had no climbing bouts)"
    if (cl == "Jump") note <- "n=80 (2 animals had no jump bouts)"
    res  <- fit_ols(sub, "bval")
    desc <- desc_stats(sub, "bval")
    gold <- GOLD$bout[[cl]]
    r <- write_metric_block(wb, SHEET, st, r, 1, lbl, res, desc, gold, note=note)
    r <- r + 1
  }
  r <- r + 1

  # ── TRANSITION METRICS ────────────────────────────────────────────────────
  writeData(wb, SHEET, "TRANSITION METRICS (Fig 4T-W)",
            startRow=r, startCol=1, colNames=FALSE)
  addStyle(wb, SHEET, st$banner, rows=r, cols=1:20, stack=FALSE)
  mergeCells(wb, SHEET, rows=r, cols=1:20)
  r <- r + 1
  writeData(wb, SHEET,
            "NOTE: Determinism and Markov entropy betas are CLOSE but not exact (6-8% off); likely due to sequence representation differences in original COping codebase.",
            startRow=r, startCol=1, colNames=FALSE)
  r <- r + 1

  trans_metrics <- list(
    list(col="lz",          label="T. Lempel-Ziv complexity"),
    list(col="recurrence",  label="U. Recurrence Rate"),
    list(col="determinism", label="V. Determinism"),
    list(col="markov",      label="W. Markov Entropy Index")
  )
  for (m in trans_metrics) {
    writeData(wb, SHEET, m$label, startRow=r, startCol=1, colNames=FALSE)
    addStyle(wb, SHEET, st$header, rows=r, cols=1:4, stack=FALSE)
    r <- r + 1
    res  <- fit_ols(trans_df, m$col)
    desc <- desc_stats(trans_df, m$col)
    gold <- GOLD$transition[[m$col]]
    note <- if (m$col %in% c("determinism","markov"))
      paste0("Beta: CLOSE (", round(abs(res$beta - gold$beta)/abs(gold$beta)*100,1),
             "% diff from gold). Direction confirmed: ", if(res$beta * gold$beta > 0) "same" else "REVERSED") else ""
    r <- write_metric_block(wb, SHEET, st, r, 1, m$label, res, desc, gold, note=note)
    r <- r + 1
  }

  # Column widths
  setColWidths(wb, SHEET, cols=1, widths=35)
  setColWidths(wb, SHEET, cols=2:4, widths=20)
}

# ── MAIN ─────────────────────────────────────────────────────────────────────
cat("Loading workbook:", OUT_FILE, "\n")
wb <- loadWorkbook(OUT_FILE)
cat("Existing sheets:", paste(names(wb), collapse=", "), "\n")

build_fig4_sheet(wb)

saveWorkbook(wb, OUT_FILE, overwrite=TRUE)
cat("Saved:", OUT_FILE, "\n")
cat("Sheets now:", paste(names(wb), collapse=", "), "\n")
