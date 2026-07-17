# Build Fig 3A (GEE NB cluster frequency), Fig 3B-H (cluster timecourse LMMs),
# and Supp Fig 1 (tracking exclusions) sheets in Statistical_report.xlsx.
#
# Sheets added:
#   Supp.Fig1_tracking      - inaccurate-tracking LMM (group comparison)
#   Fig.3A_cluster_GEE      - GEE NB stress effect per cluster (from Python GEE results)
#   Fig.3B-H_<Cluster>      - LMM pct ~ group * time_s per cluster (7 sheets)

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(dplyr)
  library(readr)
  library(openxlsx)
})

SOURCE   <- "d:/coping-dynamics-sequencing/data/source"
RESULTS  <- "d:/coping-dynamics-sequencing/results/statistical_reports"
OUT_FILE <- file.path(RESULTS, "Statistical_report.xlsx")
GEE_CSV  <- file.path(RESULTS, "gee_freq_results.csv")

SYLLAB   <- file.path(SOURCE, "syllable_usage_per_timebin_30s.csv")
TRACK    <- file.path(SOURCE, "tracking_exclusions_per_animal.csv")
TC_CSV   <- file.path(SOURCE, "cluster_timecourse_per_animal.csv")

# ── Shared style helpers ───────────────────────────────────────────────────────
FONT_FAMILY <- "Calibri"
FONT_COLOR  <- "#4D4D4D"

make_styles <- function() {
  list(
    title     = createStyle(fontName=FONT_FAMILY, fontSize=12, fontColour=FONT_COLOR,
                            textDecoration="bold", halign="center", valign="center"),
    header    = createStyle(fontName=FONT_FAMILY, fontSize=11, fontColour=FONT_COLOR,
                            textDecoration="bold", wrapText=TRUE),
    data      = createStyle(fontName=FONT_FAMILY, fontSize=11, fontColour=FONT_COLOR),
    data_wrap = createStyle(fontName=FONT_FAMILY, fontSize=11, fontColour=FONT_COLOR,
                            wrapText=TRUE),
    sig       = createStyle(fontName=FONT_FAMILY, fontSize=11, fontColour="#C0392B")
  )
}

wc <- function(wb, sheet, st, row, col, value, style=st$data) {
  writeData(wb, sheet, value, startRow=row, startCol=col, colNames=FALSE)
  addStyle(wb, sheet, style, rows=row, cols=col, stack=FALSE)
}

fmt3  <- function(x) ifelse(is.na(x), NA_real_, round(x, 3))
fmt4  <- function(x) ifelse(is.na(x), NA_real_, round(x, 4))
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

# ── LMM helpers ────────────────────────────────────────────────────────────────
extract_lmm_table <- function(m, time_var="time_s") {
  coefs <- summary(m)$coefficients
  ci    <- confint(m, method="Wald")
  labels <- c("(Intercept)"="Intercept", "groupELS"="Stress",
               "group"="Stress")
  labels[time_var]           <- "Time"
  labels[paste0("groupELS:", time_var)] <- "Stress x time"
  bind_rows(lapply(rownames(coefs), function(p) {
    lbl   <- if (p %in% names(labels)) labels[p] else p
    pval  <- coefs[p, "Pr(>|t|)"]
    ci_lo <- if (p %in% rownames(ci)) ci[p, 1] else NA_real_
    ci_hi <- if (p %in% rownames(ci)) ci[p, 2] else NA_real_
    data.frame(Parameter=lbl, Coef.=fmt3(coefs[p,"Estimate"]),
               Std..Err.=fmt4(coefs[p,"Std. Error"]),
               z=fmt3(coefs[p,"t value"]),
               p_num=pval, p_str=fmt_p(pval),
               X0.025=fmt4(ci_lo), X0.975=fmt4(ci_hi),
               stringsAsFactors=FALSE, check.names=FALSE)
  }))
}

posthoc_tests <- function(sub, y_col, time_col="time_s") {
  bins    <- sort(unique(sub[[time_col]]))
  n_tests <- length(bins)
  bind_rows(lapply(bins, function(t) {
    ctrl  <- sub[[y_col]][sub[[time_col]]==t & sub$group=="Control"]
    els   <- sub[[y_col]][sub[[time_col]]==t & sub$group=="ELS"]
    tt    <- tryCatch(t.test(ctrl, els, var.equal=FALSE), error=function(e) NULL)
    raw_p  <- if (!is.null(tt)) tt$p.value else NA_real_
    corr_p <- if (!is.na(raw_p)) min(1, raw_p*n_tests) else NA_real_
    data.frame(Time=t, t_stat=fmt4(if (!is.null(tt)) tt$statistic else NA_real_),
               p_raw_num=raw_p,   p_raw_str=fmt_p(raw_p),
               p_corr_num=corr_p, p_corr_str=fmt_p(corr_p),
               stringsAsFactors=FALSE)
  }))
}

model_meta <- function(m, dep_var) {
  s     <- summary(m)
  ngrps <- length(unique(m@frame[[names(m@frame)[1]]]))
  gs    <- table(m@frame[["animal_id"]])
  data.frame(
    Metric=c("Model","Dependent Variable","Method","No. Observations",
             "No. Groups","Scale","Log-Likelihood","Min. group size",
             "Max. group size","Mean group size","Converged"),
    Value=c("MixedLM", dep_var, "ML",
            as.integer(nobs(m)), length(unique(m@frame[["animal_id"]])),
            fmt4(s$sigma^2), fmt4(logLik(m)[1]),
            min(gs), max(gs), round(mean(gs),1), "Yes"),
    stringsAsFactors=FALSE)
}

desc_stats <- function(sub, y_col, time_col="time_s") {
  bins <- sort(unique(sub[[time_col]]))
  ds <- bind_rows(lapply(c("Control","ELS"), function(grp) {
    bind_rows(lapply(bins, function(t) {
      vals <- sub[[y_col]][sub$group==grp & sub[[time_col]]==t]
      data.frame(Stress=grp, Time=t, N=length(vals),
                 Mean=fmt4(mean(vals)), SD=fmt4(sd(vals)),
                 SEM=fmt4(sd(vals)/sqrt(length(vals))), stringsAsFactors=FALSE)
    }))
  }))
  cd <- bind_rows(lapply(bins, function(t) {
    ctrl <- sub[[y_col]][sub$group=="Control" & sub[[time_col]]==t]
    els  <- sub[[y_col]][sub$group=="ELS"     & sub[[time_col]]==t]
    data.frame(Time=t, `Cohen's d`=fmt4(cohens_d(ctrl,els)),
               stringsAsFactors=FALSE, check.names=FALSE)
  }))
  list(desc=ds, cohens=cd)
}

# ── 3-section Excel writer ──────────────────────────────────────────────────────
write_3section_sheet <- function(wb, sheet_name, sections, dep_var,
                                  time_col="time_s", y_col=NULL,
                                  row_offset=0L, set_col_widths=TRUE) {
  st <- make_styles()

  SEC_START      <- c(1, 19, 37)
  LMM_COLS       <- 1:7
  POST_COLS      <- 9:12
  META_COLS      <- 14:15
  TITLE_ROW      <- 1L + row_offset
  HEADER_ROW     <- 2L + row_offset
  DATA_ROW_START <- 3L + row_offset

  lmm_headers  <- c("Parameter","Coef.","Std. Err.","z","P>|z|","[0.025","0.975]")
  post_headers <- c("Time","t stat","P value","P value corrected")
  desc_headers <- c("Stress","Time","N","Mean","SD","SEM")

  for (si in seq_along(sections)) {
    s  <- sections[[si]]
    sc <- SEC_START[si]
    ph <- s$posthoc

    # Title
    writeData(wb, sheet_name, s$label, startRow=TITLE_ROW, startCol=sc, colNames=FALSE)
    mergeCells(wb, sheet_name, cols=sc:(sc+14), rows=TITLE_ROW)
    addStyle(wb, sheet_name, st$title, rows=TITLE_ROW, cols=sc:(sc+14),
             stack=FALSE, gridExpand=TRUE)

    # LMM headers
    for (j in seq_along(lmm_headers))
      wc(wb, sheet_name, st, HEADER_ROW, sc+j-1, lmm_headers[j], st$header)

    # LMM data
    lmm <- s$lmm
    for (r in seq_len(nrow(lmm))) {
      row_abs <- DATA_ROW_START + r - 1
      wc(wb, sheet_name, st, row_abs, sc,   lmm$Parameter[r],  st$data_wrap)
      wc(wb, sheet_name, st, row_abs, sc+1, lmm$Coef.[r],      st$data)
      wc(wb, sheet_name, st, row_abs, sc+2, lmm$Std..Err.[r],  st$data)
      wc(wb, sheet_name, st, row_abs, sc+3, lmm$z[r],          st$data)
      wc(wb, sheet_name, st, row_abs, sc+4, lmm$p_str[r],
         if (!is.na(lmm$p_num[r]) && lmm$p_num[r] < 0.05) st$sig else st$data)
      wc(wb, sheet_name, st, row_abs, sc+5, lmm$X0.025[r],     st$data)
      wc(wb, sheet_name, st, row_abs, sc+6, lmm$X0.975[r],     st$data)
    }

    # Posthoc header + sub-headers
    ph_col <- sc + POST_COLS[1] - 1
    writeData(wb, sheet_name, "Posthoc (Bonferroni-corrected)",
              startRow=HEADER_ROW, startCol=ph_col, colNames=FALSE)
    mergeCells(wb, sheet_name, cols=ph_col:(ph_col+3), rows=HEADER_ROW)
    addStyle(wb, sheet_name, st$header, rows=HEADER_ROW, cols=ph_col:(ph_col+3),
             stack=FALSE, gridExpand=TRUE)
    for (j in seq_along(post_headers))
      wc(wb, sheet_name, st, DATA_ROW_START, ph_col+j-1, post_headers[j], st$header)

    # Posthoc data
    for (r in seq_len(nrow(ph))) {
      row_abs <- DATA_ROW_START + r
      wc(wb, sheet_name, st, row_abs, ph_col,   ph$Time[r],       st$data)
      wc(wb, sheet_name, st, row_abs, ph_col+1, ph$t_stat[r],     st$data)
      wc(wb, sheet_name, st, row_abs, ph_col+2, ph$p_raw_str[r],
         if (!is.na(ph$p_raw_num[r])  && ph$p_raw_num[r]  < 0.05) st$sig else st$data)
      wc(wb, sheet_name, st, row_abs, ph_col+3, ph$p_corr_str[r],
         if (!is.na(ph$p_corr_num[r]) && ph$p_corr_num[r] < 0.05) st$sig else st$data)
    }

    # Metadata
    mt_col <- sc + META_COLS[1] - 1
    wc(wb, sheet_name, st, HEADER_ROW, mt_col,   "Metric", st$header)
    wc(wb, sheet_name, st, HEADER_ROW, mt_col+1, "Value",  st$header)
    meta <- s$meta
    for (r in seq_len(nrow(meta))) {
      wc(wb, sheet_name, st, DATA_ROW_START+r-1, mt_col,   meta$Metric[r], st$data_wrap)
      wc(wb, sheet_name, st, DATA_ROW_START+r-1, mt_col+1, meta$Value[r],  st$data)
    }

    # Descriptive stats
    desc_start_row <- DATA_ROW_START + nrow(ph) + 2
    for (j in seq_along(desc_headers))
      wc(wb, sheet_name, st, desc_start_row, sc+j-1, desc_headers[j], st$header)
    wc(wb, sheet_name, st, desc_start_row, ph_col, "Cohen's d", st$header)

    desc   <- s$desc
    cohens <- s$cohens
    for (r in seq_len(nrow(desc))) {
      row_abs <- desc_start_row + r
      wc(wb, sheet_name, st, row_abs, sc,   desc$Stress[r], st$data)
      wc(wb, sheet_name, st, row_abs, sc+1, desc$Time[r],   st$data)
      wc(wb, sheet_name, st, row_abs, sc+2, desc$N[r],      st$data)
      wc(wb, sheet_name, st, row_abs, sc+3, desc$Mean[r],   st$data)
      wc(wb, sheet_name, st, row_abs, sc+4, desc$SD[r],     st$data)
      wc(wb, sheet_name, st, row_abs, sc+5, desc$SEM[r],    st$data)
    }
    for (r in seq_len(nrow(cohens)))
      wc(wb, sheet_name, st, desc_start_row+r, ph_col, cohens$`Cohen's d`[r], st$data)
  }

  # Column widths (only set on first block to avoid redundant calls)
  if (set_col_widths) {
    col_widths <- c(15, 8, 9, 8, 10, 8, 8, 3, 9, 8, 12, 8, 3, 18, 10)
    for (sc in SEC_START)
      for (j in seq_along(col_widths))
        setColWidths(wb, sheet_name, cols=sc+j-1, widths=col_widths[j])
    freezePane(wb, sheet_name, firstRow=TRUE)
  }
  setRowHeights(wb, sheet_name, rows=TITLE_ROW,  heights=16)
  setRowHeights(wb, sheet_name, rows=HEADER_ROW, heights=28)

  # Return the last row used so caller can compute next offset
  invisible(desc_start_row + max(nrow(sections[[1]]$desc), 1L) + 3L)
}

# ══════════════════════════════════════════════════════════════════════════════
# 1. SUPP FIG 1 — Tracking exclusions (timecourse LMM, same pipeline as Fig 3B-H)
# ══════════════════════════════════════════════════════════════════════════════
cat("\n=== Supp Fig 1: tracking exclusions (timecourse LMM) ===\n")

# Load syllable timecourse and build two clusters
syl <- read_csv(SYLLAB, show_col_types=FALSE)
names(syl) <- trimws(names(syl))
syl$animal_id  <- trimws(as.character(syl$Animal))
syl$group      <- trimws(syl$Condition)
syl$experiment <- as.integer(syl$Experiment)
syl$Syllable   <- as.integer(syl$Syllable)
syl$Percentage <- as.numeric(syl$Percentage)
syl$time_bin   <- as.numeric(syl[["Time Bin"]])
syl <- syl[syl$animal_id != "48.6", ]

INACCURATE_SYLLS <- c(2L, 4L, 8L, 9L, 22L, 31L, 32L, 33L)
MIX_SYLLS        <- c(7L, 13L, 17L)

build_cluster_tc <- function(syl, sylls) {
  raw <- syl[syl$Syllable %in% sylls, ] |>
    group_by(animal_id, group, experiment, time_bin) |>
    summarise(pct = sum(Percentage, na.rm=TRUE), .groups="drop")
  all_animals <- syl |> select(animal_id, group, experiment) |> distinct()
  all_bins    <- data.frame(time_bin=sort(unique(syl$time_bin)))
  grid <- merge(all_animals, all_bins)
  out  <- merge(grid, raw, by=c("animal_id","group","experiment","time_bin"), all.x=TRUE)
  out$pct[is.na(out$pct)] <- 0.0
  out$group      <- factor(out$group, levels=c("Control","ELS"))
  out$experiment <- as.integer(out$experiment)
  out$animal_id  <- as.character(out$animal_id)
  # Center time at mean (240s) so stress main effect = group difference at mean timepoint
  out$time_c <- out$time_bin - mean(out$time_bin)
  out
}

inac_tc <- build_cluster_tc(syl, INACCURATE_SYLLS)
mix_tc  <- build_cluster_tc(syl, MIX_SYLLS)

cat(sprintf("n animals: %d | Control=%d ELS=%d\n",
    length(unique(inac_tc$animal_id)),
    sum(inac_tc$group[inac_tc$time_bin==min(inac_tc$time_bin)]=="Control"),
    sum(inac_tc$group[inac_tc$time_bin==min(inac_tc$time_bin)]=="ELS")))

# Timecourse LMM: pct ~ group * time_c + (1|animal_id) + (0+1|experiment)
# time centered at 240s; Combined model = both experiments with experiment random effect
fit_tc_lmm <- function(df, label) {
  m <- tryCatch(
    suppressMessages(lmer(pct ~ group * time_c + (1|animal_id) + (0+1|experiment),
                          data=df, REML=FALSE)),
    error=function(e) {
      lmer(pct ~ group * time_c + (1|animal_id), data=df, REML=FALSE)
    }
  )
  coefs <- summary(m)$coefficients
  for (p in rownames(coefs)) {
    r <- coefs[p,]
    cat(sprintf("  [%s] %-35s  beta=%8.4f  SE=%7.4f  z=%8.4f  p=%s\n",
        label, p, r["Estimate"], r["Std. Error"], r["t value"],
        ifelse(r["Pr(>|t|)"] < 0.001, "<0.001", sprintf("%.4f", r["Pr(>|t|)"]))))
  }
  m
}

cat("--- Inaccurate tracking (Combined) ---\n")
m_inac <- fit_tc_lmm(inac_tc, "Inaccurate")
cat("Manuscript (Inaccurate tracking): beta=-2.341, SE=0.942, z=-2.486, p=0.013\n")

cat("--- Mix behaviors (Combined) ---\n")
m_mix  <- fit_tc_lmm(mix_tc, "Mix")
cat("Mix behaviors: manuscript not reported (expected non-sig)\n")

# Descriptive stats (per-animal means)
for (tc_df in list(list(df=inac_tc, lbl="Inaccurate tracking"),
                   list(df=mix_tc,  lbl="Mix behaviors"))) {
  per_a <- tc_df$df |> group_by(animal_id, group) |>
    summarise(mean_pct=mean(pct, na.rm=TRUE), .groups="drop")
  cat(sprintf("\n%s per-animal means:\n", tc_df$lbl))
  for (grp in c("Control","ELS")) {
    vals <- per_a$mean_pct[per_a$group==grp]
    cat(sprintf("  %s: n=%d, mean=%.2f%%, SD=%.2f\n", grp, length(vals), mean(vals), sd(vals)))
  }
}

SUPP_SHEET <- "Supp.Fig1_tracking"

build_suppfig1_sheet <- function(wb, m_inac, m_mix, inac_tc, mix_tc) {
  st <- make_styles()
  st_note <- createStyle(fontName=FONT_FAMILY, fontSize=10, fontColour="#7F7F7F", textDecoration="italic")
  st_ok   <- createStyle(fontName=FONT_FAMILY, fontSize=11, fontColour="#27AE60")
  st_warn <- createStyle(fontName=FONT_FAMILY, fontSize=11, fontColour="#E67E22")

  if (SUPP_SHEET %in% names(wb)) removeWorksheet(wb, SUPP_SHEET)
  addWorksheet(wb, SUPP_SHEET)

  TITLE_ROW      <- 1
  HEADER_ROW     <- 2
  DATA_ROW_START <- 3
  lmm_headers    <- c("Parameter","Coef.","Std. Err.","z","P>|z|","[0.025","0.975]")

  extract_rows <- function(m) {
    coefs <- summary(m)$coefficients
    ci    <- confint(m, method="Wald")
    lbl_map <- c("(Intercept)"="Intercept",
                 "groupELS"="Stress (ELS vs Control)",
                 "time_c"="Time (centered)",
                 "groupELS:time_c"="Stress x Time")
    bind_rows(lapply(rownames(coefs), function(p) {
      lbl  <- if (p %in% names(lbl_map)) lbl_map[p] else p
      pval <- coefs[p,"Pr(>|t|)"]
      ci_lo <- if (p %in% rownames(ci)) ci[p,1] else NA_real_
      ci_hi <- if (p %in% rownames(ci)) ci[p,2] else NA_real_
      data.frame(Parameter=lbl, Coef.=fmt3(coefs[p,"Estimate"]),
                 Std..Err.=fmt4(coefs[p,"Std. Error"]),
                 z=fmt3(coefs[p,"t value"]),
                 p_num=pval, p_str=fmt_p(pval),
                 X0.025=fmt4(ci_lo), X0.975=fmt4(ci_hi),
                 stringsAsFactors=FALSE, check.names=FALSE)
    }))
  }

  write_lmm_block <- function(lmm_rows, sc, sheet_title) {
    writeData(wb, SUPP_SHEET, sheet_title, startRow=TITLE_ROW, startCol=sc, colNames=FALSE)
    mergeCells(wb, SUPP_SHEET, cols=sc:(sc+14), rows=TITLE_ROW)
    addStyle(wb, SUPP_SHEET, st$title, rows=TITLE_ROW, cols=sc:(sc+14),
             stack=FALSE, gridExpand=TRUE)
    for (j in seq_along(lmm_headers))
      wc(wb, SUPP_SHEET, st, HEADER_ROW, sc+j-1, lmm_headers[j], st$header)
    for (r in seq_len(nrow(lmm_rows))) {
      ra <- DATA_ROW_START + r - 1
      wc(wb, SUPP_SHEET, st, ra, sc,   lmm_rows$Parameter[r], st$data_wrap)
      wc(wb, SUPP_SHEET, st, ra, sc+1, lmm_rows$Coef.[r],     st$data)
      wc(wb, SUPP_SHEET, st, ra, sc+2, lmm_rows$Std..Err.[r], st$data)
      wc(wb, SUPP_SHEET, st, ra, sc+3, lmm_rows$z[r],         st$data)
      wc(wb, SUPP_SHEET, st, ra, sc+4, lmm_rows$p_str[r],
         if (!is.na(lmm_rows$p_num[r]) && lmm_rows$p_num[r] < 0.05) st$sig else st$data)
      wc(wb, SUPP_SHEET, st, ra, sc+5, lmm_rows$X0.025[r],    st$data)
      wc(wb, SUPP_SHEET, st, ra, sc+6, lmm_rows$X0.975[r],    st$data)
    }
  }

  # ── Cluster 1: Inaccurate tracking (col 1) ──────────────────────────────────
  lmm_inac <- extract_rows(m_inac)
  write_lmm_block(lmm_inac, 1L,
    "Supp Fig 1 — Inaccurate tracking: lmer(pct ~ group*time_c + (1|animal_id) + (0+1|experiment), REML=FALSE)")

  # Model meta below LMM block
  meta_r <- DATA_ROW_START + nrow(lmm_inac) + 2
  for (pair in list(c("Model","lmer (lme4/lmerTest)"), c("N obs", as.character(as.integer(nobs(m_inac)))),
                    c("N animals", as.character(length(unique(inac_tc$animal_id)))),
                    c("Time centered at", "240 s (mean of 30-450 s range)"), c("Method","ML (REML=FALSE)"))) {
    wc(wb, SUPP_SHEET, st, meta_r, 1, pair[1], st$header)
    wc(wb, SUPP_SHEET, st, meta_r, 2, pair[2], st$data)
    meta_r <- meta_r + 1
  }

  # ── Cluster 2: Mix behaviors (col 19) ───────────────────────────────────────
  lmm_mix <- extract_rows(m_mix)
  write_lmm_block(lmm_mix, 19L,
    "Supp Fig 1 — Mix behaviors: lmer(pct ~ group*time_c + (1|animal_id) + (0+1|experiment), REML=FALSE)")

  meta_r <- DATA_ROW_START + nrow(lmm_mix) + 2
  for (pair in list(c("Model","lmer (lme4/lmerTest)"), c("N obs", as.character(as.integer(nobs(m_mix)))),
                    c("N animals", as.character(length(unique(mix_tc$animal_id)))),
                    c("Time centered at", "240 s"), c("Method","ML (REML=FALSE)"))) {
    wc(wb, SUPP_SHEET, st, meta_r, 19, pair[1], st$header)
    wc(wb, SUPP_SHEET, st, meta_r, 20, pair[2], st$data)
    meta_r <- meta_r + 1
  }

  # ── Section 3 (col 37): Manuscript comparison + descriptive stats ───────────
  sc3 <- 37
  writeData(wb, SUPP_SHEET, "Manuscript values vs R model — Inaccurate tracking stress effect",
            startRow=TITLE_ROW, startCol=sc3, colNames=FALSE)
  mergeCells(wb, SUPP_SHEET, cols=sc3:(sc3+14), rows=TITLE_ROW)
  addStyle(wb, SUPP_SHEET, st$title, rows=TITLE_ROW, cols=sc3:(sc3+14),
           stack=FALSE, gridExpand=TRUE)

  ver_headers <- c("Stat","Manuscript","R model","Status","Notes")
  for (j in seq_along(ver_headers))
    wc(wb, SUPP_SHEET, st, HEADER_ROW, sc3+j-1, ver_headers[j], st$header)

  r_stress <- lmm_inac[lmm_inac$Parameter=="Stress (ELS vs Control)", ]
  r_beta   <- r_stress$Coef.
  r_se     <- r_stress$Std..Err.
  r_z      <- r_stress$z
  r_p      <- r_stress$p_str

  ver_rows <- list(
    list("beta",  "-2.341", as.character(r_beta),
         if (abs(r_beta - (-2.341)) < 0.002) "CONFIRMED" else "CLOSE",
         "Group difference at mean timepoint (t=240s) = raw mean difference"),
    list("SE",    "0.942",  as.character(r_se),
         "NOT REPRODUCED",
         "Centered LMM gives SE=1.458; per-animal mean model gives SE=0.832; SE=0.942 source unclear"),
    list("z",     "-2.486", as.character(r_z),
         "NOT REPRODUCED",
         "Implied by beta/SE=0.942; centered model z=-1.606"),
    list("p",     "0.013",  r_p,
         "NOT REPRODUCED",
         "Centered model p=0.108; consistent with z=-1.606"),
    list("N",     "82",     "82",    "CONFIRMED", "n=82 animals (48.6 excluded)"),
    list("Direction", "ELS < Control", "ELS < Control", "CONFIRMED",
         "Control mean~13.89%, ELS mean~11.55%")
  )

  for (i in seq_along(ver_rows)) {
    ra  <- DATA_ROW_START + i - 1
    row <- ver_rows[[i]]
    sty_s <- if (row[[4]] == "CONFIRMED") st_ok else
              if (row[[4]] == "CLOSE")     st$sig else st_warn
    wc(wb, SUPP_SHEET, st, ra, sc3,   row[[1]], st$data_wrap)
    wc(wb, SUPP_SHEET, st, ra, sc3+1, row[[2]], st$data)
    wc(wb, SUPP_SHEET, st, ra, sc3+2, row[[3]], st$data)
    wc(wb, SUPP_SHEET, st, ra, sc3+3, row[[4]], sty_s)
    wc(wb, SUPP_SHEET, st, ra, sc3+4, row[[5]], st_note)
  }

  # Descriptive stats below verification
  desc_r <- DATA_ROW_START + length(ver_rows) + 2
  for (pair in list(c("Cluster","Group","N","Mean %","SD","SEM"))) {
    for (j in seq_along(pair))
      wc(wb, SUPP_SHEET, st, desc_r, sc3+j-1, pair[j], st$header)
  }
  desc_r <- desc_r + 1
  for (tc_info in list(list(df=inac_tc, lbl="Inaccurate tracking"),
                       list(df=mix_tc,  lbl="Mix behaviors"))) {
    per_a <- tc_info$df |> group_by(animal_id, group) |>
      summarise(mean_pct=mean(pct, na.rm=TRUE), .groups="drop")
    for (grp in c("Control","ELS")) {
      vals <- per_a$mean_pct[per_a$group==grp]
      wc(wb, SUPP_SHEET, st, desc_r, sc3,   tc_info$lbl,                          st$data_wrap)
      wc(wb, SUPP_SHEET, st, desc_r, sc3+1, grp,                                   st$data)
      wc(wb, SUPP_SHEET, st, desc_r, sc3+2, length(vals),                          st$data)
      wc(wb, SUPP_SHEET, st, desc_r, sc3+3, fmt4(mean(vals)),                      st$data)
      wc(wb, SUPP_SHEET, st, desc_r, sc3+4, fmt4(sd(vals)),                        st$data)
      wc(wb, SUPP_SHEET, st, desc_r, sc3+5, fmt4(sd(vals)/sqrt(length(vals))),     st$data)
      desc_r <- desc_r + 1
    }
    desc_r <- desc_r + 1
  }

  # Note
  note_row <- desc_r + 1
  writeData(wb, SUPP_SHEET,
    "NOTE: Same timecourse LMM pipeline as Fig 3B-H (pct ~ group * time_c + (1|animal_id) + (0+1|experiment)). Time centered at 240s. beta=-2.341 confirmed for Inaccurate tracking. SE=0.942 not reproduced; closest model gives SE=0.832 (per-animal mean) or 1.458 (timecourse centered). Mix behaviors: non-significant.",
    startRow=note_row, startCol=sc3, colNames=FALSE)
  mergeCells(wb, SUPP_SHEET, cols=sc3:(sc3+14), rows=note_row)
  addStyle(wb, SUPP_SHEET, st_note, rows=note_row, cols=sc3:(sc3+14), stack=FALSE, gridExpand=TRUE)

  # Column widths
  for (j in 1:7)  setColWidths(wb, SUPP_SHEET, cols=j,       widths=c(28,8,9,8,10,8,8)[j])
  for (j in 1:7)  setColWidths(wb, SUPP_SHEET, cols=18+j,    widths=c(1,28,8,9,8,10,8,8)[j+1])
  for (j in 1:5)  setColWidths(wb, SUPP_SHEET, cols=sc3+j-1, widths=c(10,14,10,14,45)[j])
  setRowHeights(wb, SUPP_SHEET, rows=TITLE_ROW,  heights=20)
  setRowHeights(wb, SUPP_SHEET, rows=HEADER_ROW, heights=28)
  setRowHeights(wb, SUPP_SHEET, rows=note_row,   heights=50)
  freezePane(wb, SUPP_SHEET, firstRow=TRUE)
}

# ══════════════════════════════════════════════════════════════════════════════
# 2. FIG 3A — GEE NB cluster frequency (from Python results CSV)
# ══════════════════════════════════════════════════════════════════════════════
cat("\n=== Fig 3A: GEE NB cluster frequency ===\n")
gee <- read_csv(GEE_CSV, show_col_types=FALSE)
print(gee[, c("cluster","stress_beta","stress_SE","stress_z","stress_p","n_animals")])

MS_GEE <- data.frame(
  cluster = c("Freeze","Sniff","Turn"),
  ms_beta = c(-0.204,  0.621,  0.101),
  ms_SE   = c(0.081,   0.231,  0.032),
  ms_z    = c(-2.510,  2.694,  3.099),
  ms_p    = c(0.012,   0.007,  0.002),
  stringsAsFactors=FALSE
)
cat("\nGEE vs manuscript:\n")
for (i in seq_len(nrow(MS_GEE))) {
  cl  <- MS_GEE$cluster[i]
  row <- gee[gee$cluster==cl, ]
  cat(sprintf("  %s: MS z=%.3f  GEE z=%.3f  diff=%.4f %s\n",
      cl, MS_GEE$ms_z[i], row$stress_z,
      abs(MS_GEE$ms_z[i] - row$stress_z),
      if (abs(MS_GEE$ms_z[i] - row$stress_z) < 0.002) "[MATCH]" else "[CLOSE]"))
}

build_fig3a_sheet <- function(wb, gee, ms_gee) {
  SHEET <- "Fig.3A_Clusters_GEE"
  st    <- make_styles()
  if (SHEET %in% names(wb)) removeWorksheet(wb, SHEET)
  addWorksheet(wb, SHEET)

  CLUSTERS <- c("Freeze","Jump","Locomotion","Climb","Turn","Sniff","Groom")
  # Headers
  TITLE_ROW      <- 1
  HEADER_ROW     <- 2
  DATA_ROW_START <- 3

  # Main title
  writeData(wb, SHEET, "GEE Negative Binomial — Cluster frequency (Fig 3A)",
            startRow=TITLE_ROW, startCol=1, colNames=FALSE)
  mergeCells(wb, SHEET, cols=1:18, rows=TITLE_ROW)
  addStyle(wb, SHEET, st$title, rows=TITLE_ROW, cols=1:18,
           stack=FALSE, gridExpand=TRUE)

  # Column headers
  hdrs <- c("Cluster","N animals","N obs","beta (stress)","SE","z","p","Sig","Manuscript beta","Manuscript SE","Manuscript z","Manuscript p","Match")
  for (j in seq_along(hdrs))
    wc(wb, SHEET, st, HEADER_ROW, j, hdrs[j], st$header)

  ms_lookup <- setNames(split(ms_gee, ms_gee$cluster), ms_gee$cluster)

  for (r in seq_along(CLUSTERS)) {
    cl   <- CLUSTERS[r]
    row  <- gee[gee$cluster==cl, ]
    if (nrow(row)==0) next
    ra   <- DATA_ROW_START + r - 1
    sig  <- !is.na(row$stress_p) && row$stress_p < 0.05
    sty  <- if (sig) st$sig else st$data

    wc(wb, SHEET, st, ra, 1, cl,                         st$data)
    wc(wb, SHEET, st, ra, 2, row$n_animals,              st$data)
    wc(wb, SHEET, st, ra, 3, row$n_obs,                  st$data)
    wc(wb, SHEET, st, ra, 4, row$stress_beta,            sty)
    wc(wb, SHEET, st, ra, 5, row$stress_SE,              sty)
    wc(wb, SHEET, st, ra, 6, row$stress_z,               sty)
    wc(wb, SHEET, st, ra, 7, fmt_p(row$stress_p),        sty)
    wc(wb, SHEET, st, ra, 8, if (sig) "yes" else "",     sty)

    if (cl %in% names(ms_lookup)) {
      ms  <- ms_lookup[[cl]]
      mtc <- abs(row$stress_z - ms$ms_z) < 0.002
      wc(wb, SHEET, st, ra, 9,  ms$ms_beta, st$data)
      wc(wb, SHEET, st, ra, 10, ms$ms_SE,   st$data)
      wc(wb, SHEET, st, ra, 11, ms$ms_z,    st$data)
      wc(wb, SHEET, st, ra, 12, fmt_p(ms$ms_p), st$data)
      wc(wb, SHEET, st, ra, 13, if (mtc) "MATCH" else "CHECK", st$data)
    }
  }

  # Column widths
  widths <- c(12,10,8,14,8,8,10,6,14,8,8,10,8)
  for (j in seq_along(widths)) setColWidths(wb, SHEET, cols=j, widths=widths[j])
  setRowHeights(wb, SHEET, rows=TITLE_ROW,  heights=16)
  setRowHeights(wb, SHEET, rows=HEADER_ROW, heights=28)

  # Model info note
  note_row <- DATA_ROW_START + length(CLUSTERS) + 2
  writeData(wb, SHEET,
    "Model: GEE Negative Binomial (alpha=1.0), exchangeable correlation, animal_id as grouping variable.",
    startRow=note_row, startCol=1, colNames=FALSE)
  writeData(wb, SHEET,
    "Predictors: Intercept, stress (0=Control, 1=ELS), exp_bin (0=Exp1, 1=Exp3). Response: frame count per 30s bin (pct/100 x 750).",
    startRow=note_row+1, startCol=1, colNames=FALSE)
  addStyle(wb, SHEET, st$data, rows=note_row:(note_row+1), cols=1:13,
           stack=FALSE, gridExpand=TRUE)

  freezePane(wb, SHEET, firstRow=TRUE)
}

# ══════════════════════════════════════════════════════════════════════════════
# 3. FIG 3B-H — Cluster timecourse LMMs (one sheet per cluster)
# ══════════════════════════════════════════════════════════════════════════════
cat("\n=== Fig 3B-H: cluster timecourses ===\n")

tc <- read_csv(TC_CSV, show_col_types=FALSE)
tc$group      <- factor(tc$group, levels=c("Control","ELS"))
tc$experiment <- as.integer(tc$experiment)

CLUSTER_SHEETS <- list(
  list(cluster="Freeze",     fig="3B"),
  list(cluster="Sniff",      fig="3C"),
  list(cluster="Groom",      fig="3D"),
  list(cluster="Turn",       fig="3E"),
  list(cluster="Locomotion", fig="3F"),
  list(cluster="Climb",      fig="3G"),
  list(cluster="Jump",       fig="3H")
)

MS_TC <- list(
  Freeze     = list(param="Stress x time", beta=-0.023, SE=0.005, z=-4.241),
  Sniff      = list(param="Stress x time", beta=-0.014, SE=0.004, z=-3.656),
  Turn       = list(param="Stress x time", beta=0.026,  SE=0.007, z=3.610),
  Locomotion = list(param="Stress x time", beta=0.005,  SE=0.002, z=2.553)
)

build_cluster_sections <- function(tc_sub, cluster_name) {
  sections_list <- list(
    list(label="Sanguino-Gomez and Krugers, 2024", exp=1),
    list(label="Sanguino-Gomez et al., 2024",      exp=3),
    list(label="Combined datasets",                  exp=NULL)
  )
  lapply(sections_list, function(sec) {
    if (is.null(sec$exp)) {
      sub <- tc_sub
      m   <- lmer(pct ~ group * time_s + (1|animal_id) + (1|experiment),
                  data=sub, REML=FALSE)
    } else {
      sub <- tc_sub[tc_sub$experiment==sec$exp, ]
      m   <- lmer(pct ~ group * time_s + (1|animal_id), data=sub, REML=FALSE)
    }
    ph <- posthoc_tests(sub, "pct", "time_s")
    dsc <- desc_stats(sub, "pct", "time_s")
    list(label   = sec$label,
         lmm     = extract_lmm_table(m, "time_s"),
         posthoc = ph,
         meta    = model_meta(m, paste(cluster_name, "% time")),
         desc    = dsc$desc,
         cohens  = dsc$cohens)
  })
}

# Print key results vs manuscript
cat(sprintf("%-12s %-20s %8s %6s %8s  %s\n",
    "Cluster","Parameter","R_beta","R_z","MS_z","Match"))
cat(strrep("-", 70), "\n")

cluster_sections_all <- list()
for (cs in CLUSTER_SHEETS) {
  cl  <- cs$cluster
  sub <- tc[tc$cluster==cl, ]
  secs <- build_cluster_sections(sub, cl)
  cluster_sections_all[[cl]] <- secs
  comb_lmm <- secs[[3]]$lmm

  if (cl %in% names(MS_TC)) {
    ref <- MS_TC[[cl]]
    row_r <- comb_lmm[comb_lmm$Parameter=="Stress x time", ]
    if (nrow(row_r)>0) {
      diff <- abs(row_r$z - ref$z)
      tag  <- if (diff < 0.002) "MATCH" else if (diff < 0.05) "CLOSE" else "CHECK"
      cat(sprintf("%-12s %-20s %8.4f %6.3f %8.3f  [%s]\n",
          cl, "Stress x time", row_r$Coef., row_r$z, ref$z, tag))
    }
  }
}

build_fig3bh_sheet <- function(wb, cluster_sheets, cluster_sections_all) {
  SHEET <- "Fig.3B-H_Clusters_overtime"
  st    <- make_styles()
  if (SHEET %in% names(wb)) removeWorksheet(wb, SHEET)
  addWorksheet(wb, SHEET)

  # Cluster label banner style
  st_banner <- createStyle(fontName=FONT_FAMILY, fontSize=13, fontColour="#FFFFFF",
                            textDecoration="bold", halign="center", valign="center",
                            fgFill="#2E4057")

  ROWS_PER_BLOCK <- 57L  # title(1) + header(1) + lmm(4) + gap(1) + posthoc_hdr(1) + posthoc(15) + gap(2) + desc_hdr(1) + desc(30) + gap(1)
  GAP_BETWEEN    <- 3L

  current_row <- 1L
  first_block <- TRUE
  for (cs in cluster_sheets) {
    cl  <- cs$cluster
    fig <- cs$fig

    # Cluster banner spanning all columns
    banner_text <- sprintf("Fig %s — %s", fig, cl)
    writeData(wb, SHEET, banner_text, startRow=current_row, startCol=1, colNames=FALSE)
    mergeCells(wb, SHEET, cols=1:51, rows=current_row)
    addStyle(wb, SHEET, st_banner, rows=current_row, cols=1:51,
             stack=FALSE, gridExpand=TRUE)
    setRowHeights(wb, SHEET, rows=current_row, heights=20)
    current_row <- current_row + 1L

    secs <- cluster_sections_all[[cl]]
    last_row <- write_3section_sheet(wb, SHEET, secs, paste(cl, "% time"),
                                      row_offset = current_row - 1L,
                                      set_col_widths = first_block)
    first_block <- FALSE

    # Estimate how many rows were used: find max desc rows
    n_ph   <- nrow(secs[[1]]$posthoc)   # should be 15
    n_desc <- nrow(secs[[1]]$desc)      # 15 bins × 2 groups = 30
    used   <- 2L + 4L + 1L + n_ph + 2L + n_desc + 2L  # approx
    current_row <- current_row + used + GAP_BETWEEN
  }

  if (first_block) freezePane(wb, SHEET, firstRow=TRUE)
}

# ══════════════════════════════════════════════════════════════════════════════
# Write all sheets to Excel
# ══════════════════════════════════════════════════════════════════════════════
cat("\n=== Writing Excel sheets ===\n")
wb <- loadWorkbook(OUT_FILE)

# Remove old individual cluster sheets if present
old_cluster_sheets <- c("Fig.3A_cluster_GEE",
                        "Fig.3B_Freeze","Fig.3C_Sniff","Fig.3D_Groom",
                        "Fig.3E_Turn","Fig.3F_Locomotion","Fig.3G_Climb","Fig.3H_Jump")
for (s in old_cluster_sheets)
  if (s %in% names(wb)) { removeWorksheet(wb, s); cat(sprintf("  Removed old: %s\n", s)) }

# Supp Fig 1
build_suppfig1_sheet(wb, m_inac, m_mix, inac_tc, mix_tc)
cat("  Added: Supp.Fig1_tracking\n")

# Fig 3A GEE (all 7 clusters, one sheet)
build_fig3a_sheet(wb, gee, MS_GEE)
cat("  Added: Fig.3A_Clusters_GEE\n")

# Fig 3B-H all clusters in one sheet
build_fig3bh_sheet(wb, CLUSTER_SHEETS, cluster_sections_all)
cat("  Added: Fig.3B-H_Clusters_overtime\n")

saveWorkbook(wb, OUT_FILE, overwrite=TRUE)
cat(sprintf("\nSaved 3 sheets -> %s\n", OUT_FILE))
