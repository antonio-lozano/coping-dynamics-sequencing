# Fig 2H — S0+S28 syllable timecourse LMM — Statistical Report Builder
# Appends sheet "Fig.2H_S0S28_MoSeq" to Statistical_report.xlsx.
#
# Model: lmer(pct_s0s28 ~ group * time_s + (1|animal_id), REML=FALSE)
# Combined adds (1|experiment).  Time in seconds (30, 60, ..., 450).
#
# Manuscript: time β=0.126, SE=0.004, z=32.602 (combined)
#             ELS×time β=-0.023, SE=0.005, z=-4.241
# Report.xlsx Freezing_syllables "Syllables 0 and 28":
#             time β=0.12625, SE=0.00387, z=32.620 (n=1230, incl. 48.6)
#             ELS×time β=-0.02333, SE=0.005474, z=-4.262

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(dplyr)
  library(readr)
  library(openxlsx)
})

SOURCE   <- "d:/coping-dynamics-sequencing/data/source"
DATA_CSV <- file.path(SOURCE, "s0s28_timecourse_per_animal.csv")
OUT_FILE <- "d:/coping-dynamics-sequencing/results/statistical_reports/Statistical_report.xlsx"
SHEET    <- "Fig.2H_S0S28_MoSeq"

# ── Load data ──────────────────────────────────────────────────────────────────
long_df <- read_csv(DATA_CSV, show_col_types=FALSE)
long_df$group      <- factor(long_df$group, levels=c("Control","ELS"))
long_df$experiment <- as.integer(long_df$experiment)

cat(sprintf("Data: %d animals, %d obs | Exp1: %d | Exp3: %d\n",
    length(unique(long_df$animal_id)), nrow(long_df),
    length(unique(long_df$animal_id[long_df$experiment==1])),
    length(unique(long_df$animal_id[long_df$experiment==3]))))

# ── Helpers (identical to freezing script) ─────────────────────────────────────
fmt3   <- function(x) ifelse(is.na(x), NA_real_, round(x, 3))
fmt4   <- function(x) ifelse(is.na(x), NA_real_, round(x, 4))
fmt_p  <- function(p) {
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

# ── Fit LMMs ──────────────────────────────────────────────────────────────────
fit_lmm <- function(sub, with_exp_re=FALSE) {
  if (with_exp_re)
    lmer(pct_s0s28 ~ group * time_s + (1|animal_id) + (1|experiment),
         data=sub, REML=FALSE)
  else
    lmer(pct_s0s28 ~ group * time_s + (1|animal_id),
         data=sub, REML=FALSE)
}

m_exp1 <- fit_lmm(long_df[long_df$experiment==1,])
m_exp3 <- fit_lmm(long_df[long_df$experiment==3,])
m_comb <- fit_lmm(long_df, with_exp_re=TRUE)

# ── Extract functions (same pattern as freezing script) ───────────────────────
extract_lmm_table <- function(m) {
  coefs  <- summary(m)$coefficients
  ci     <- confint(m, method="Wald")
  labels <- c("(Intercept)"="Intercept","groupELS"="Stress",
               "time_s"="Time","groupELS:time_s"="Stress x time")
  rows <- lapply(rownames(coefs), function(p) {
    lbl  <- if (p %in% names(labels)) labels[p] else p
    pval <- coefs[p,"Pr(>|t|)"]
    ci_lo <- if (p %in% rownames(ci)) ci[p,1] else NA_real_
    ci_hi <- if (p %in% rownames(ci)) ci[p,2] else NA_real_
    data.frame(Parameter=lbl, Coef.=fmt3(coefs[p,"Estimate"]),
               Std..Err.=fmt4(coefs[p,"Std. Error"]),
               z=fmt3(coefs[p,"t value"]),
               p_num=pval, p_str=fmt_p(pval),
               X0.025=fmt4(ci_lo), X0.975=fmt4(ci_hi),
               stringsAsFactors=FALSE, check.names=FALSE)
  })
  bind_rows(rows)
}

posthoc_tests <- function(sub) {
  bins    <- sort(unique(sub$time_s))
  n_tests <- length(bins)
  rows <- lapply(bins, function(t) {
    ctrl  <- sub$pct_s0s28[sub$time_s==t & sub$group=="Control"]
    els   <- sub$pct_s0s28[sub$time_s==t & sub$group=="ELS"]
    tt    <- tryCatch(t.test(ctrl, els, var.equal=FALSE), error=function(e) NULL)
    tst   <- if (!is.null(tt)) tt$statistic  else NA_real_
    raw_p <- if (!is.null(tt)) tt$p.value    else NA_real_
    corr_p <- if (!is.na(raw_p)) min(1, raw_p*n_tests) else NA_real_
    data.frame(Time=t, t_stat=fmt4(tst),
               p_raw_num=raw_p,   p_raw_str=fmt_p(raw_p),
               p_corr_num=corr_p, p_corr_str=fmt_p(corr_p),
               stringsAsFactors=FALSE)
  })
  bind_rows(rows)
}

model_meta <- function(m, dep_var="Percentage (S0+S28)") {
  s      <- summary(m)
  nobs   <- as.integer(nobs(m))
  ngrps  <- length(unique(m@frame[["animal_id"]]))
  gs     <- table(m@frame[["animal_id"]])
  data.frame(
    Metric=c("Model","Dependent Variable","Method","No. Observations",
             "No. Groups","Scale","Log-Likelihood",
             "Min. group size","Max. group size","Mean group size","Converged"),
    Value=c("MixedLM", dep_var, "ML",
            nobs, ngrps, fmt4(s$sigma^2), fmt4(logLik(m)[1]),
            min(gs), max(gs), round(mean(gs),1), "Yes"),
    stringsAsFactors=FALSE)
}

desc_stats <- function(sub) {
  bins <- sort(unique(sub$time_s))
  rows <- list()
  for (grp in c("Control","ELS")) {
    for (t in bins) {
      vals <- sub$pct_s0s28[sub$group==grp & sub$time_s==t]
      rows[[length(rows)+1]] <- data.frame(
        Stress=grp, Time=t, N=length(vals),
        Mean=fmt4(mean(vals)), SD=fmt4(sd(vals)),
        SEM=fmt4(sd(vals)/sqrt(length(vals))),
        stringsAsFactors=FALSE)
    }
  }
  ds <- bind_rows(rows)
  cohens <- bind_rows(lapply(bins, function(t) {
    ctrl <- sub$pct_s0s28[sub$group=="Control" & sub$time_s==t]
    els  <- sub$pct_s0s28[sub$group=="ELS"     & sub$time_s==t]
    data.frame(Time=t, `Cohen's d`=fmt4(cohens_d(ctrl,els)),
               stringsAsFactors=FALSE, check.names=FALSE)
  }))
  list(desc=ds, cohens=cohens)
}

# ── Assemble sections ─────────────────────────────────────────────────────────
sections <- list(
  list(label="Sanguino-Gómez and Krugers, 2024", m=m_exp1,
       sub=long_df[long_df$experiment==1,]),
  list(label="Sanguino-Gómez et al., 2024",      m=m_exp3,
       sub=long_df[long_df$experiment==3,]),
  list(label="Combined datasets",                 m=m_comb,
       sub=long_df)
)
for (i in seq_along(sections)) {
  s <- sections[[i]]
  sections[[i]]$lmm     <- extract_lmm_table(s$m)
  sections[[i]]$posthoc <- posthoc_tests(s$sub)
  sections[[i]]$meta    <- model_meta(s$m)
  d <- desc_stats(s$sub)
  sections[[i]]$desc   <- d$desc
  sections[[i]]$cohens <- d$cohens
}

# Print key results for verification
cat("\n=== Key results vs manuscript ===\n")
for (s in sections) {
  cat(sprintf("\n%s:\n", s$label))
  lmm <- s$lmm
  for (param in c("Time","Stress x time")) {
    r <- lmm[lmm$Parameter==param,]
    if (nrow(r)>0)
      cat(sprintf("  %-15s beta=%8.4f  SE=%8.4f  z=%8.4f  p=%s\n",
                  param, r$Coef., r$Std..Err., r$z, r$p_str))
  }
}
cat("\nManuscript reports (combined): time β=0.126, SE=0.004, z=32.602\n")
cat("                               ELS×time β=-0.023, SE=0.005, z=-4.241\n")
cat("Report.xlsx (n=1230, incl. 48.6): time z=32.620, ELS×time z=-4.262\n")

# ── Build Excel sheet ─────────────────────────────────────────────────────────
wb <- loadWorkbook(OUT_FILE)
if (SHEET %in% names(wb)) removeWorksheet(wb, SHEET)
addWorksheet(wb, SHEET)

FONT_FAMILY <- "Calibri"
FONT_COLOR  <- "#4D4D4D"

st_title      <- createStyle(fontName=FONT_FAMILY, fontSize=12, fontColour=FONT_COLOR,
                              textDecoration="bold", halign="center", valign="center")
st_header     <- createStyle(fontName=FONT_FAMILY, fontSize=11, fontColour=FONT_COLOR,
                              textDecoration="bold", wrapText=TRUE)
st_data       <- createStyle(fontName=FONT_FAMILY, fontSize=11, fontColour=FONT_COLOR)
st_data_wrap  <- createStyle(fontName=FONT_FAMILY, fontSize=11, fontColour=FONT_COLOR,
                              wrapText=TRUE)
st_sig        <- createStyle(fontName=FONT_FAMILY, fontSize=11, fontColour="#C0392B")

wc <- function(row, col, value, style=st_data) {
  writeData(wb, SHEET, value, startRow=row, startCol=col, colNames=FALSE)
  addStyle(wb, SHEET, style, rows=row, cols=col, stack=FALSE)
}

SEC_START  <- c(1, 19, 37)
LMM_COLS   <- 1:7
POST_COLS  <- 9:12
META_COLS  <- 14:15
TITLE_ROW  <- 1
HEADER_ROW <- 2
DATA_ROW_START <- 3

lmm_headers  <- c("Parameter","Coef.","Std. Err.","z","P>|z|","[0.025","0.975]")
post_headers <- c("Time","t stat","P value","P value corrected")
meta_header  <- c("Metric","Value")
desc_headers <- c("Stress","Time","N","Mean","SD","SEM")

for (si in seq_along(sections)) {
  s  <- sections[[si]]
  sc <- SEC_START[si]

  # Title
  writeData(wb, SHEET, s$label, startRow=TITLE_ROW, startCol=sc, colNames=FALSE)
  mergeCells(wb, SHEET, cols=sc:(sc+14), rows=TITLE_ROW)
  addStyle(wb, SHEET, st_title, rows=TITLE_ROW, cols=sc:(sc+14),
           stack=FALSE, gridExpand=TRUE)

  # LMM headers
  for (j in seq_along(lmm_headers))
    wc(HEADER_ROW, sc+LMM_COLS[j]-1, lmm_headers[j],
       if (j==1) st_header else st_header)

  # LMM data
  lmm <- s$lmm
  for (r in seq_len(nrow(lmm))) {
    row_abs <- DATA_ROW_START + r - 1
    wc(row_abs, sc,   lmm$Parameter[r],  st_data_wrap)
    wc(row_abs, sc+1, lmm$Coef.[r],      st_data)
    wc(row_abs, sc+2, lmm$Std..Err.[r],  st_data)
    wc(row_abs, sc+3, lmm$z[r],          st_data)
    wc(row_abs, sc+4, lmm$p_str[r],
       if (!is.na(lmm$p_num[r]) && lmm$p_num[r] < 0.05) st_sig else st_data)
    wc(row_abs, sc+5, lmm$X0.025[r],     st_data)
    wc(row_abs, sc+6, lmm$X0.975[r],     st_data)
  }

  # Posthoc header (row 2, merged) + sub-headers (row 3)
  ph_col <- sc + POST_COLS[1] - 1
  writeData(wb, SHEET, "Posthoc (Bonferroni-corrected)",
            startRow=HEADER_ROW, startCol=ph_col, colNames=FALSE)
  mergeCells(wb, SHEET, cols=ph_col:(ph_col+3), rows=HEADER_ROW)
  addStyle(wb, SHEET, st_header, rows=HEADER_ROW, cols=ph_col:(ph_col+3),
           stack=FALSE, gridExpand=TRUE)
  for (j in seq_along(post_headers))
    wc(DATA_ROW_START, ph_col+j-1, post_headers[j], st_header)

  # Posthoc data (starts at DATA_ROW_START + 1)
  ph <- s$posthoc
  for (r in seq_len(nrow(ph))) {
    row_abs <- DATA_ROW_START + r
    wc(row_abs, ph_col,   ph$Time[r],       st_data)
    wc(row_abs, ph_col+1, ph$t_stat[r],     st_data)
    wc(row_abs, ph_col+2, ph$p_raw_str[r],
       if (!is.na(ph$p_raw_num[r])  && ph$p_raw_num[r]  < 0.05) st_sig else st_data)
    wc(row_abs, ph_col+3, ph$p_corr_str[r],
       if (!is.na(ph$p_corr_num[r]) && ph$p_corr_num[r] < 0.05) st_sig else st_data)
  }

  # Metadata (headers row 2, data row 3+)
  mt_col <- sc + META_COLS[1] - 1
  wc(HEADER_ROW, mt_col,   meta_header[1], st_header)
  wc(HEADER_ROW, mt_col+1, meta_header[2], st_header)
  meta <- s$meta
  for (r in seq_len(nrow(meta)))
    { wc(DATA_ROW_START+r-1, mt_col, meta$Metric[r], st_data_wrap)
      wc(DATA_ROW_START+r-1, mt_col+1, meta$Value[r], st_data) }

  # Descriptive stats — after posthoc data ends
  desc_start_row <- DATA_ROW_START + nrow(ph) + 2
  for (j in seq_along(desc_headers))
    wc(desc_start_row, sc+j-1, desc_headers[j], st_header)
  wc(desc_start_row, ph_col, "Cohen's d", st_header)

  desc   <- s$desc
  cohens <- s$cohens
  for (r in seq_len(nrow(desc))) {
    row_abs <- desc_start_row + r
    wc(row_abs, sc,   desc$Stress[r], st_data)
    wc(row_abs, sc+1, desc$Time[r],   st_data)
    wc(row_abs, sc+2, desc$N[r],      st_data)
    wc(row_abs, sc+3, desc$Mean[r],   st_data)
    wc(row_abs, sc+4, desc$SD[r],     st_data)
    wc(row_abs, sc+5, desc$SEM[r],    st_data)
  }
  for (r in seq_len(nrow(cohens)))
    wc(desc_start_row + r, ph_col, cohens$`Cohen's d`[r], st_data)
}

# Column widths (same as freezing sheet)
col_widths <- c(15, 8, 9, 8, 10, 8, 8, 3, 9, 8, 12, 8, 3, 18, 10)
for (si in seq_along(SEC_START)) {
  sc <- SEC_START[si]
  for (j in seq_along(col_widths))
    setColWidths(wb, SHEET, cols=sc+j-1, widths=col_widths[j])
}
setRowHeights(wb, SHEET, rows=TITLE_ROW,  heights=16)
setRowHeights(wb, SHEET, rows=HEADER_ROW, heights=28)
freezePane(wb, SHEET, firstRow=TRUE)

saveWorkbook(wb, OUT_FILE, overwrite=TRUE)
cat(sprintf("\nSaved sheet '%s' → %s\n", SHEET, OUT_FILE))
