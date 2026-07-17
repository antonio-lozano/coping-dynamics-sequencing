#!/usr/bin/env Rscript
# Statistical analysis for Sanguino-Gomez et al. coping dynamics manuscript
# Models:
#   - LMM (lme4/lmerTest): continuous outcomes, random intercepts for animal + experiment
#   - GEE Negative Binomial (geepack): discrete/count transition data

suppressPackageStartupMessages({
  library(readxl)
  library(lme4)
  library(lmerTest)
  library(geepack)
  library(MASS)
  library(dplyr)
  library(tidyr)
})

RDATA  <- "D:/coping-dynamics-sequencing/results/figure_data/Raw_data.xlsx"
OUTDIR <- "D:/coping-dynamics-sequencing/results/statistical_reports"
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)

BIN_SECONDS <- 30

# ── helpers ──────────────────────────────────────────────────────────────────

read_sheet <- function(path, sheet) {
  raw <- read_excel(path, sheet = sheet, col_names = FALSE)
  # row 0 = title (merged), row 1 = headers, row 2+ = data
  headers <- as.character(unlist(raw[2, ]))
  df <- raw[-(1:2), ]
  colnames(df) <- headers
  df <- type.convert(df, as.is = TRUE)
  as.data.frame(df)
}

lmm_summary <- function(model, label, effect) {
  cf <- summary(model)$coefficients
  data.frame(
    label     = label,
    effect    = rownames(cf),
    beta      = cf[, "Estimate"],
    SE        = cf[, "Std. Error"],
    df        = if ("df" %in% colnames(cf)) cf[, "df"] else NA_real_,
    t_or_z    = if ("t value" %in% colnames(cf)) cf[, "t value"] else cf[, "z value"],
    p_value   = if ("Pr(>|t|)" %in% colnames(cf)) cf[, "Pr(>|t|)"] else cf[, "Pr(>|z|)"],
    row.names = NULL
  )
}

gee_summary <- function(model, label) {
  cf <- summary(model)$coefficients
  data.frame(
    label   = label,
    effect  = rownames(cf),
    beta    = cf[, "Estimate"],
    SE      = cf[, "Std.err"],
    t_or_z  = cf[, "Wald"],
    p_value = cf[, "Pr(>|W|)"],
    row.names = NULL
  )
}

wide_to_long <- function(df, id_cols, time_cols, value_name = "pct") {
  df_long <- pivot_longer(df,
    cols      = all_of(time_cols),
    names_to  = "time_label",
    values_to = value_name
  )
  df_long <- df_long[!is.na(df_long[[value_name]]), ]
  # extract seconds from label like "t_030s"
  df_long$time_s <- as.numeric(sub("t_(\\d+)s", "\\1", df_long$time_label))
  df_long
}

results_list <- list()

# ── Fig 2H: S0+S28 timecourse LMM ────────────────────────────────────────────
# Uses precomputed fixed-bin CSV (moseq_syllables_per_frame.csv.gz source, 81 animals × 15 bins)
cat("\n=== Fig 2H: S0+S28 usage timecourse ===\n")

s028_long <- read.csv("D:/coping-dynamics-sequencing/data/source/s0s28_timecourse_per_animal.csv")
s028_long$stress     <- ifelse(s028_long$group == "ELS", 1L, 0L)
s028_long$animal_id  <- as.factor(s028_long$animal_id)
# experiment column is integer (1 or 3) — convert to factor
s028_long$experiment <- as.factor(s028_long$experiment)

cat("n obs =", nrow(s028_long), "| n animals =", nlevels(s028_long$animal_id),
    "| bins =", length(unique(s028_long$bin)), "\n")

m_s028 <- lmer(pct_s0s28 ~ time_s * stress + (1 | animal_id) + (1 | experiment),
               data = s028_long, REML = FALSE)
print(summary(m_s028)$coefficients)
results_list[["Fig2H_S0S28_timecourse"]] <- lmm_summary(m_s028, "Fig2H_S0S28_timecourse", "LMM")


# ── Fig 3A-C: Supervised freezing timecourse LMM ─────────────────────────────
cat("\n=== Fig 3A-C: Supervised freezing timecourse ===\n")

frz <- read_sheet(RDATA, "Freezing_combined")
time_cols_frz <- grep("^t_\\d+s$", colnames(frz), value = TRUE)
frz_long <- wide_to_long(frz, id_cols = c("animal_id","group","experiment"), time_cols = time_cols_frz)
frz_long$stress     <- ifelse(frz_long$group == "ELS", 1L, 0L)
frz_long$animal_id  <- as.factor(frz_long$animal_id)
frz_long$experiment <- as.factor(frz_long$experiment)

bin_counts_f  <- table(frz_long$animal_id)
common_bins_f <- as.numeric(names(sort(table(bin_counts_f), decreasing = TRUE)[1]))
keep_times_f  <- sort(unique(frz_long$time_s))[1:common_bins_f]
frz_long      <- frz_long[frz_long$time_s %in% keep_times_f, ]

cat("n obs =", nrow(frz_long), "| n animals =", nlevels(frz_long$animal_id), "\n")
m_frz <- lmer(pct ~ time_s * stress + (1 | animal_id) + (1 | experiment),
              data = frz_long, REML = FALSE)
print(summary(m_frz)$coefficients)
results_list[["Fig3A_freezing_timecourse"]] <- lmm_summary(m_frz, "Fig3A_freezing_timecourse", "LMM")


# ── Fig 3 cluster frequency LMM ──────────────────────────────────────────────
# Cross-sectional (one obs per animal) → random intercept for experiment only
cat("\n=== Fig 3 cluster frequency ===\n")

clust_freq <- read_sheet(RDATA, "Cluster_frequency")
clust_freq$stress     <- ifelse(clust_freq$group == "ELS", 1L, 0L)
clust_freq$animal_id  <- as.factor(clust_freq$animal_id)
clust_freq$experiment <- as.factor(clust_freq$experiment)
clust_freq$frequency_seconds <- as.numeric(clust_freq$frequency_seconds)

for (cl in unique(clust_freq$cluster)) {
  sub <- clust_freq[clust_freq$cluster == cl, ]
  lbl <- paste0("Fig3_freq_", cl)
  tryCatch({
    m <- lmer(frequency_seconds ~ stress + (1 | experiment),
              data = sub, REML = FALSE)
    cat(cl, ":\n")
    print(summary(m)$coefficients)
    results_list[[lbl]] <- lmm_summary(m, lbl, "LMM")
  }, error = function(e) cat(cl, "failed:", conditionMessage(e), "\n"))
}


# ── Fig 3 cluster timecourse LMM ─────────────────────────────────────────────
cat("\n=== Fig 3 cluster timecourse ===\n")

clust_time <- read_sheet(RDATA, "Cluster_time_wide")
time_cols_c <- grep("^t_\\d+s$", colnames(clust_time), value = TRUE)

for (cl in unique(clust_time$cluster)) {
  sub <- clust_time[clust_time$cluster == cl, ]
  sub_long <- wide_to_long(sub,
    id_cols   = c("animal_id","group","experiment"),
    time_cols = time_cols_c
  )
  sub_long$stress     <- ifelse(sub_long$group == "ELS", 1L, 0L)
  sub_long$animal_id  <- as.factor(sub_long$animal_id)
  sub_long$experiment <- as.factor(sub_long$experiment)
  sub_long <- sub_long[!is.na(sub_long$pct), ]

  bc  <- table(sub_long$animal_id)
  cbn <- as.numeric(names(sort(table(bc), decreasing = TRUE)[1]))
  kt  <- sort(unique(sub_long$time_s))[1:cbn]
  sub_long <- sub_long[sub_long$time_s %in% kt, ]

  lbl <- paste0("Fig3_timecourse_", cl)
  tryCatch({
    m <- lmer(pct ~ time_s * stress + (1 | animal_id) + (1 | experiment),
              data = sub_long, REML = FALSE)
    cat(cl, ":\n")
    print(summary(m)$coefficients)
    results_list[[lbl]] <- lmm_summary(m, lbl, "LMM")
  }, error = function(e) cat(cl, "failed:", conditionMessage(e), "\n"))
}


# ── Fig 4: Diversity / transition metrics LMM ────────────────────────────────
cat("\n=== Fig 4: Diversity metrics ===\n")

div <- read_sheet(RDATA, "Fig4_frequency_metrics")
div$stress     <- ifelse(div$group == "ELS", 1L, 0L)
div$animal_id  <- as.factor(div$animal_id)
div$experiment <- as.factor(div$experiment)

for (metric in c("simpson_index","shannon_entropy_index","evenness_index","cumulative_usage_index")) {
  div[[metric]] <- as.numeric(div[[metric]])
  lbl <- paste0("Fig4_diversity_", metric)
  tryCatch({
    m <- lmer(as.formula(paste0(metric, " ~ stress + (1 | experiment)")),
              data = div, REML = FALSE)
    cat(metric, ":\n")
    print(summary(m)$coefficients)
    results_list[[lbl]] <- lmm_summary(m, lbl, "LMM")
  }, error = function(e) cat(metric, "failed:", conditionMessage(e), "\n"))
}


cat("\n=== Fig 4: Transition metrics (continuous) LMM ===\n")

trans <- read_sheet(RDATA, "Fig4_transition_metrics")
trans$stress     <- ifelse(trans$group == "ELS", 1L, 0L)
trans$animal_id  <- as.factor(trans$animal_id)
trans$experiment <- as.factor(trans$experiment)

for (metric in c("lempel_ziv_complexity","recurrence_rate","determinism","markov_entropy")) {
  trans[[metric]] <- as.numeric(trans[[metric]])
  lbl <- paste0("Fig4_transition_", metric)
  tryCatch({
    m <- lmer(as.formula(paste0(metric, " ~ stress + (1 | experiment)")),
              data = trans, REML = FALSE)
    cat(metric, ":\n")
    print(summary(m)$coefficients)
    results_list[[lbl]] <- lmm_summary(m, lbl, "LMM")
  }, error = function(e) cat(metric, "LMM:", conditionMessage(e), "\n"))
}


cat("\n=== Fig 4: Bout duration LMM ===\n")

bouts <- read_sheet(RDATA, "Fig4_bout_duration")
bouts$stress          <- ifelse(bouts$group == "ELS", 1L, 0L)
bouts$animal_id       <- as.factor(bouts$animal_id)
bouts$experiment      <- as.factor(bouts$experiment)
bouts$bout_duration_seconds <- as.numeric(bouts$bout_duration_seconds)

for (cl in unique(bouts$cluster)) {
  sub <- bouts[bouts$cluster == cl, ]
  lbl <- paste0("Fig4_bout_duration_", cl)
  # bout_duration is per-animal mean (one row per animal per cluster)
  # → random intercept for experiment only
  tryCatch({
    m <- lmer(bout_duration_seconds ~ stress + (1 | experiment),
              data = sub, REML = FALSE)
    cat(cl, ":\n")
    print(summary(m)$coefficients)
    results_list[[lbl]] <- lmm_summary(m, lbl, "LMM")
  }, error = function(e) cat(cl, "bout failed:", conditionMessage(e), "\n"))
}


# ── Fig 5 / BFL score LMM ────────────────────────────────────────────────────
cat("\n=== Fig 5: Behavioral dynamics score ===\n")

bfl <- read_sheet(RDATA, "BFL_scores")
bfl <- bfl[!is.na(bfl$Score) & !is.na(bfl$Condition), ]
bfl$Score      <- as.numeric(bfl$Score)
bfl$stress     <- ifelse(bfl$Condition == "ELS", 1L, 0L)
bfl$animal_id  <- as.factor(bfl$Animal)
bfl$experiment <- as.factor(bfl$Experiment)

tryCatch({
  m_bfl <- lmer(Score ~ stress + (1 | experiment),
                data = bfl, REML = FALSE)
  cat("BFL score:\n")
  print(summary(m_bfl)$coefficients)
  results_list[["Fig5_BFL_score"]] <- lmm_summary(m_bfl, "Fig5_BFL_score", "LMM")
}, error = function(e) cat("BFL failed:", conditionMessage(e), "\n"))


# ── GEE NB: Transition count data ────────────────────────────────────────────
cat("\n=== GEE NB: Lempel-Ziv complexity (discrete count) ===\n")

# lz needs to be an integer count — check if values from source_data are counts
# The Fig4_transition_metrics has normalized lz; we need the raw count.
# From source_data_figure4.csv: mean lz Control=191.85, ELS=183.85 (look like integer counts)
# But per-animal raw counts aren't in Raw_data.xlsx.
# We use the continuous lz from Fig4_transition_metrics with GEE as a robustness check.

trans_gee <- trans
trans_gee$exp_id <- as.numeric(factor(trans_gee$experiment))

tryCatch({
  gee_lz <- geeglm(
    round(lempel_ziv_complexity * 100) ~ stress,
    data   = trans_gee,
    id     = exp_id,
    family = poisson(link = "log"),
    corstr = "exchangeable"
  )
  cat("GEE Poisson lz proxy:\n")
  print(summary(gee_lz)$coefficients)
  results_list[["GEE_lz_proxy"]] <- gee_summary(gee_lz, "GEE_lz_proxy")
}, error = function(e) cat("GEE lz failed:", conditionMessage(e), "\n"))


# ── Save all results ──────────────────────────────────────────────────────────
cat("\n=== Saving results ===\n")
all_results <- bind_rows(results_list)
out_csv <- file.path(OUTDIR, "lmm_results.csv")
write.csv(all_results, out_csv, row.names = FALSE)
cat("Saved:", out_csv, "\n")
cat("Total model results:", nrow(all_results), "rows\n")

# Print summary table
cat("\n=== KEY RESULTS SUMMARY ===\n")
key_effects <- all_results[all_results$effect %in% c("time_s","stress","time_s:stress"), ]
print(key_effects[, c("label","effect","beta","SE","t_or_z","p_value")])
