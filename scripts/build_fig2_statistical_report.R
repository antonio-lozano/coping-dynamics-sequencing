# Fig 2A-C Freezing LMM — Statistical Report Builder
# Runs mixed-effects models (lme4, REML=FALSE) on SimBA freezing data and
# writes results/statistical_reports/Fig2_Freezing_LMM.xlsx matching the
# aesthetics of the lab Report.xlsx template.
#
# Output sheet layout (3 sections side-by-side, each 15 cols wide, 3-col gap):
#   Section 1 (A-O):  Sanguino-Gómez and Krugers, 2024  (Exp 1)
#   Section 2 (S-AG): Sanguino-Gómez et al., 2024       (Exp 3)
#   Section 3 (AK-AY): Combined datasets
#
# Each section:
#   A-G  : LMM parameter table  (7 cols)
#   H    : gap
#   I-L  : Per-timebin post-hoc Welch t-tests, Bonferroni-corrected (4 cols)
#   M    : gap
#   N-O  : Model metadata  (2 cols)
# Then descriptive stats table below (Stress / Time / N / Mean / SD / SEM + Cohen's d)

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(dplyr)
  library(readr)
  library(openxlsx)
})

# ── Paths ─────────────────────────────────────────────────────────────────────
SOURCE       <- "d:/coping-dynamics-sequencing/data/source"
FREEZING_DIR <- file.path(SOURCE, "freezing_predictions")
INDEX_CSV    <- file.path(SOURCE, "animal_groups.csv")
FREQ_CSV     <- file.path(SOURCE, "cluster_frequency_per_animal.csv")
OUT_DIR      <- "d:/coping-dynamics-sequencing/results/statistical_reports"
OUT_FILE     <- file.path(OUT_DIR, "Statistical_report.xlsx")

BIN_SECONDS  <- 30
FPS          <- 25      # actual recording FPS (11250 frames / 7.5 min / 60 s = 25 fps)
EXCLUDE      <- c("animal_48_6", "48_6")

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# ── Helpers ───────────────────────────────────────────────────────────────────
normalize_name <- function(x) gsub(" ", "_", trimws(x))

short_id <- function(x) {
  norm <- normalize_name(x)
  m <- regmatches(norm, regexpr("(?i)Animal[_ ]?(\\d+(?:[_-]\\d+)?)", norm, perl=TRUE))
  if (length(m) > 0 && nchar(m) > 0) {
    id <- sub("(?i)Animal[_ ]?", "", m[1], perl=TRUE)
    return(gsub("[_-]", ".", id))
  }
  parts <- strsplit(norm, "_")[[1]]
  n <- length(parts)
  if (n >= 2) paste(parts[(n-1):n], collapse=".") else norm
}

cohens_d <- function(x, y) {
  nx <- length(x); ny <- length(y)
  if (nx < 2 || ny < 2) return(NA_real_)
  sp <- sqrt(((nx-1)*var(x) + (ny-1)*var(y)) / (nx+ny-2))
  if (sp == 0) return(NA_real_)
  (mean(x) - mean(y)) / sp
}

fmt3 <- function(x) ifelse(is.na(x), NA_real_, round(x, 3))
fmt4 <- function(x) ifelse(is.na(x), NA_real_, round(x, 4))

# ── Load data ─────────────────────────────────────────────────────────────────
idx       <- read_csv(INDEX_CSV, show_col_types=FALSE)
group_map <- setNames(idx$group, normalize_name(idx$name))
group_sid <- setNames(idx$group, sapply(idx$name, short_id))

freq    <- read_csv(FREQ_CSV, show_col_types=FALSE)
exp_map <- setNames(freq$experiment, as.character(freq$animal_id))

bin_size <- FPS * BIN_SECONDS   # 750 frames
csvs <- list.files(FREEZING_DIR, pattern="_freezing_predictions_only\\.csv$",
                   full.names=TRUE)

records <- list()
for (f in csvs) {
  base <- sub("_freezing_predictions_only\\.csv$", "", basename(f))
  if (tolower(normalize_name(base)) %in% EXCLUDE) next
  df <- tryCatch(read_csv(f, show_col_types=FALSE, progress=FALSE), error=function(e) NULL)
  if (is.null(df)) next
  if (!"Freezing_Jen_0-125_threshold" %in% names(df)) next
  sid  <- short_id(base)
  norm <- normalize_name(base)
  grp  <- group_map[norm]
  if (is.na(grp)) grp <- group_sid[sid]
  if (is.na(grp) || !grp %in% c("Control", "ELS")) next
  freeze  <- df[["Freezing_Jen_0-125_threshold"]]
  n_bins  <- floor(length(freeze) / bin_size)
  if (n_bins == 0) next
  for (b in seq_len(n_bins)) {
    chunk <- freeze[((b-1)*bin_size + 1):(b*bin_size)]
    records[[length(records)+1]] <- data.frame(
      animal_id    = sid,
      group        = grp,
      time_bin     = b,
      time_s       = b * BIN_SECONDS,
      freezing_pct = mean(chunk, na.rm=TRUE) * 100,
      stringsAsFactors = FALSE
    )
  }
}
long_df <- bind_rows(records)

# Experiment assignment
long_df$exp_key    <- as.character(suppressWarnings(as.numeric(gsub("_",".",long_df$animal_id))))
long_df$experiment <- as.integer(exp_map[long_df$exp_key])
long_df <- long_df[!is.na(long_df$experiment), ]
long_df$group <- factor(long_df$group, levels=c("Control","ELS"))

cat(sprintf("Loaded: %d animals, %d obs | Exp1: %d animals | Exp3: %d animals\n",
    length(unique(long_df$animal_id)), nrow(long_df),
    length(unique(long_df$animal_id[long_df$experiment==1])),
    length(unique(long_df$animal_id[long_df$experiment==3]))))

# ── Fit LMMs ──────────────────────────────────────────────────────────────────
fit_lmm <- function(sub, with_exp_re=FALSE) {
  if (with_exp_re) {
    lmer(freezing_pct ~ group * time_bin + (1|animal_id) + (1|experiment),
         data=sub, REML=FALSE)
  } else {
    lmer(freezing_pct ~ group * time_bin + (1|animal_id),
         data=sub, REML=FALSE)
  }
}

m_exp1 <- fit_lmm(long_df[long_df$experiment==1,])
m_exp3 <- fit_lmm(long_df[long_df$experiment==3,])
m_comb <- fit_lmm(long_df, with_exp_re=TRUE)

fmt_p <- function(p) {
  if (is.na(p)) return(NA_character_)
  if (p < 0.001) return("<0.001")
  as.character(round(p, 4))
}

extract_lmm_table <- function(m) {
  coefs  <- summary(m)$coefficients
  ci     <- confint(m, method="Wald")
  params <- rownames(coefs)
  labels <- c(
    "(Intercept)"       = "Intercept",
    "groupELS"          = "Stress",
    "time_bin"          = "Time",
    "groupELS:time_bin" = "Stress x time"
  )
  rows <- lapply(params, function(p) {
    lbl   <- if (p %in% names(labels)) labels[p] else p
    pval  <- coefs[p, "Pr(>|t|)"]
    ci_lo <- if (p %in% rownames(ci)) ci[p,1] else NA_real_
    ci_hi <- if (p %in% rownames(ci)) ci[p,2] else NA_real_
    data.frame(
      Parameter  = lbl,
      Coef.      = fmt3(coefs[p,"Estimate"]),
      Std..Err.  = fmt3(coefs[p,"Std. Error"]),
      z          = fmt3(coefs[p,"t value"]),
      p_num      = pval,          # kept numeric for conditional styling
      p_str      = fmt_p(pval),   # displayed string
      X0.025     = fmt3(ci_lo),
      X0.975     = fmt3(ci_hi),
      stringsAsFactors = FALSE, check.names=FALSE
    )
  })
  bind_rows(rows)
}

# Per-timebin post-hoc Welch t-tests (Bonferroni)
posthoc_tests <- function(sub) {
  bins    <- sort(unique(sub$time_bin))
  n_tests <- length(bins)
  rows <- lapply(bins, function(b) {
    ctrl  <- sub$freezing_pct[sub$time_bin == b & sub$group == "Control"]
    els   <- sub$freezing_pct[sub$time_bin == b & sub$group == "ELS"]
    tt    <- tryCatch(t.test(ctrl, els, var.equal=FALSE), error=function(e) NULL)
    tst   <- if (!is.null(tt)) tt$statistic  else NA_real_
    raw_p <- if (!is.null(tt)) tt$p.value    else NA_real_
    corr_p <- if (!is.na(raw_p)) min(1, raw_p * n_tests) else NA_real_
    data.frame(
      Time          = b * BIN_SECONDS,
      t_stat        = fmt4(tst),
      p_raw_num     = raw_p,
      p_raw_str     = fmt_p(raw_p),
      p_corr_num    = corr_p,
      p_corr_str    = fmt_p(corr_p),
      stringsAsFactors=FALSE
    )
  })
  bind_rows(rows)
}

# Model metadata
model_meta <- function(m, dep_var="Freezing_predictions") {
  s <- summary(m)
  nobs    <- as.integer(nobs(m))
  ngroups <- length(unique(m@frame[["animal_id"]]))
  gs      <- table(m@frame[["animal_id"]])
  data.frame(
    Metric = c("Model","Dependent Variable","Method","No. Observations",
               "No. Groups","Scale","Log-Likelihood",
               "Min. group size","Max. group size","Mean group size","Converged"),
    Value  = c("MixedLM", dep_var, "ML",
               nobs, ngroups,
               fmt4(s$sigma^2),
               fmt4(logLik(m)[1]),
               min(gs), max(gs), round(mean(gs),1),
               "Yes"),
    stringsAsFactors=FALSE
  )
}

# Descriptive stats per group × timebin + Cohen's d
desc_stats <- function(sub) {
  bins <- sort(unique(sub$time_bin))
  rows <- list()
  for (grp in c("Control","ELS")) {
    for (b in bins) {
      vals <- sub$freezing_pct[sub$group == grp & sub$time_bin == b]
      rows[[length(rows)+1]] <- data.frame(
        Stress = grp, Time = b * BIN_SECONDS,
        N = length(vals), Mean = fmt4(mean(vals)), SD = fmt4(sd(vals)),
        SEM = fmt4(sd(vals)/sqrt(length(vals))),
        stringsAsFactors=FALSE
      )
    }
  }
  ds <- bind_rows(rows)
  # Cohen's d per timebin
  d_rows <- lapply(bins, function(b) {
    ctrl <- sub$freezing_pct[sub$group=="Control" & sub$time_bin==b]
    els  <- sub$freezing_pct[sub$group=="ELS"     & sub$time_bin==b]
    data.frame(Time=b*BIN_SECONDS, `Cohen's d`=fmt4(cohens_d(ctrl,els)),
               stringsAsFactors=FALSE, check.names=FALSE)
  })
  list(desc=ds, cohens=bind_rows(d_rows))
}

# ── Assemble section data ──────────────────────────────────────────────────────
sections <- list(
  list(label="Sanguino-Gómez and Krugers, 2024", m=m_exp1,
       sub=long_df[long_df$experiment==1,]),
  list(label="Sanguino-Gómez et al., 2024",      m=m_exp3,
       sub=long_df[long_df$experiment==3,]),
  list(label="Combined datasets",                     m=m_comb,
       sub=long_df)
)

for (i in seq_along(sections)) {
  s <- sections[[i]]
  sections[[i]]$lmm     <- extract_lmm_table(s$m)
  sections[[i]]$posthoc <- posthoc_tests(s$sub)
  sections[[i]]$meta    <- model_meta(s$m)
  d <- desc_stats(s$sub)
  sections[[i]]$desc    <- d$desc
  sections[[i]]$cohens  <- d$cohens
}

# ── Build Excel ───────────────────────────────────────────────────────────────
wb <- createWorkbook()
addWorksheet(wb, "Fig.2A-C_freezing_SimBA")
ws_name <- "Fig.2A-C_freezing_SimBA"

# Style definitions
FONT_COLOR  <- "#4D4D4D"
FONT_FAMILY <- "Calibri"

st_title <- createStyle(
  fontName=FONT_FAMILY, fontSize=12, fontColour=FONT_COLOR,
  textDecoration="bold", halign="center", valign="center"
)
st_header <- createStyle(
  fontName=FONT_FAMILY, fontSize=11, fontColour=FONT_COLOR,
  textDecoration="bold", wrapText=TRUE
)
st_param_header <- createStyle(
  fontName=FONT_FAMILY, fontSize=11, fontColour=FONT_COLOR,
  textDecoration="bold", wrapText=TRUE
)
st_data <- createStyle(
  fontName=FONT_FAMILY, fontSize=11, fontColour=FONT_COLOR
)
st_data_wrap <- createStyle(
  fontName=FONT_FAMILY, fontSize=11, fontColour=FONT_COLOR, wrapText=TRUE
)
st_sig <- createStyle(
  fontName=FONT_FAMILY, fontSize=11, fontColour="#C0392B"
)

# Each section starts at col: 1, 19, 37  (1-based; gap of 3 cols between sections)
# Layout within a section (col offsets, 1-based):
#   1-7:  LMM table
#   8:    gap
#   9-12: posthoc
#   13:   gap
#   14-15: metrics
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

# Helper to write a cell
wc <- function(row, col, value, style=st_data) {
  writeData(wb, ws_name, value, startRow=row, startCol=col, colNames=FALSE)
  addStyle(wb, ws_name, style, rows=row, cols=col, stack=FALSE)
}

# Helper to write a row of values
wr <- function(row, col_start, values, style=st_data) {
  for (i in seq_along(values)) {
    wc(row, col_start + i - 1, values[[i]], style)
  }
}

for (si in seq_along(sections)) {
  s    <- sections[[si]]
  sc   <- SEC_START[si]   # column offset for this section

  # ── Title (row 1, merged across all 15 cols of section) ────────────────────
  writeData(wb, ws_name, s$label, startRow=TITLE_ROW, startCol=sc, colNames=FALSE)
  mergeCells(wb, ws_name, cols=sc:(sc+14), rows=TITLE_ROW)
  addStyle(wb, ws_name, st_title, rows=TITLE_ROW, cols=sc:(sc+14), stack=FALSE, gridExpand=TRUE)

  # ── LMM section headers (row 2) ────────────────────────────────────────────
  for (j in seq_along(lmm_headers)) {
    wc(HEADER_ROW, sc + LMM_COLS[j] - 1, lmm_headers[j],
       if (j==1) st_param_header else st_header)
  }

  # ── LMM data rows ──────────────────────────────────────────────────────────
  lmm <- s$lmm
  for (r in seq_len(nrow(lmm))) {
    row_abs <- DATA_ROW_START + r - 1
    wc(row_abs, sc,   lmm$Parameter[r], st_data_wrap)
    wc(row_abs, sc+1, lmm$`Coef.`[r],    st_data)
    wc(row_abs, sc+2, lmm$Std..Err.[r], st_data)
    wc(row_abs, sc+3, lmm$z[r],         st_data)
    pval_num <- lmm$p_num[r]
    wc(row_abs, sc+4, lmm$p_str[r],
       if (!is.na(pval_num) && pval_num < 0.05) st_sig else st_data)
    wc(row_abs, sc+5, lmm$X0.025[r], st_data)
    wc(row_abs, sc+6, lmm$X0.975[r], st_data)
  }

  # ── Posthoc section ────────────────────────────────────────────────────────
  ph_col <- sc + POST_COLS[1] - 1
  # row 2: merged "Posthoc (Bonferroni-corrected)" header
  writeData(wb, ws_name, "Posthoc (Bonferroni-corrected)",
            startRow=HEADER_ROW, startCol=ph_col, colNames=FALSE)
  mergeCells(wb, ws_name, cols=ph_col:(ph_col+3), rows=HEADER_ROW)
  addStyle(wb, ws_name, st_header, rows=HEADER_ROW, cols=ph_col:(ph_col+3),
           stack=FALSE, gridExpand=TRUE)
  # row 3: sub-headers
  for (j in seq_along(post_headers)) {
    wc(DATA_ROW_START, ph_col + j - 1, post_headers[j], st_header)
  }
  # rows 4+: posthoc data (starts at DATA_ROW_START + 1 to avoid overwriting sub-headers)
  ph <- s$posthoc
  for (r in seq_len(nrow(ph))) {
    row_abs <- DATA_ROW_START + r
    wc(row_abs, ph_col,   ph$Time[r],    st_data)
    wc(row_abs, ph_col+1, ph$t_stat[r],  st_data)
    wc(row_abs, ph_col+2, ph$p_raw_str[r],
       if (!is.na(ph$p_raw_num[r])  && ph$p_raw_num[r]  < 0.05) st_sig else st_data)
    wc(row_abs, ph_col+3, ph$p_corr_str[r],
       if (!is.na(ph$p_corr_num[r]) && ph$p_corr_num[r] < 0.05) st_sig else st_data)
  }

  # ── Metadata section ───────────────────────────────────────────────────────
  mt_col <- sc + META_COLS[1] - 1
  wc(HEADER_ROW, mt_col,   meta_header[1], st_header)
  wc(HEADER_ROW, mt_col+1, meta_header[2], st_header)
  meta <- s$meta
  for (r in seq_len(nrow(meta))) {
    wc(DATA_ROW_START + r - 1, mt_col,   meta$Metric[r], st_data_wrap)
    wc(DATA_ROW_START + r - 1, mt_col+1, meta$Value[r],  st_data)
  }

  # ── Descriptive stats — must start AFTER posthoc data ends (row gap +2) ───
  # posthoc sub-headers at DATA_ROW_START, data rows DATA_ROW_START+1 .. DATA_ROW_START+nrow(ph)
  desc_start_row <- DATA_ROW_START + nrow(ph) + 2

  for (j in seq_along(desc_headers)) {
    wc(desc_start_row, sc + j - 1, desc_headers[j], st_header)
  }
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
  # Cohen's d aligns with Control rows (first half of desc, bins 1-15)
  for (r in seq_len(nrow(cohens))) {
    wc(desc_start_row + r, ph_col, cohens$`Cohen's d`[r], st_data)
  }
}

# ── Column widths ──────────────────────────────────────────────────────────────
# Section-local widths: [param, coef, se, z, p, ci_lo, ci_hi, gap, t, pval, pcorr, gap_post, metric, value]
col_widths <- c(15, 8, 9, 8, 10, 8, 8,  3,  9, 8, 12, 8, 3, 18, 10)
for (si in seq_along(SEC_START)) {
  sc <- SEC_START[si]
  for (j in seq_along(col_widths)) {
    setColWidths(wb, ws_name, cols=sc+j-1, widths=col_widths[j])
  }
}

# Row heights
setRowHeights(wb, ws_name, rows=TITLE_ROW,  heights=16)
setRowHeights(wb, ws_name, rows=HEADER_ROW, heights=28)

# Freeze pane
freezePane(wb, ws_name, firstRow=TRUE)

# ── Save ───────────────────────────────────────────────────────────────────────
saveWorkbook(wb, OUT_FILE, overwrite=TRUE)
cat(sprintf("\nSaved: %s\n", OUT_FILE))
