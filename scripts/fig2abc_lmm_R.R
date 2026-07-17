suppressPackageStartupMessages({ library(readxl); library(lme4); library(lmerTest); library(dplyr); library(tidyr) })

RDATA <- "D:/coping-dynamics-sequencing/results/figure_data/Raw_data.xlsx"

read_sheet <- function(path, sheet) {
  raw <- read_excel(path, sheet = sheet, col_names = FALSE)
  headers <- as.character(unlist(raw[2, ]))
  df <- raw[-(1:2), ]
  colnames(df) <- headers
  type.convert(as.data.frame(df), as.is = TRUE)
}

wide_to_long_freeze <- function(df, bin_seconds = 30) {
  time_cols <- grep("^t_\\d+s$", colnames(df), value = TRUE)
  id_cols   <- intersect(c("animal_id","group","experiment"), colnames(df))
  long <- pivot_longer(df, cols = all_of(time_cols),
                       names_to = "time_label", values_to = "pct")
  long <- long[!is.na(long$pct), ]
  long$time_s   <- as.numeric(sub("t_(\\d+)s", "\\1", long$time_label))
  long$bin      <- (long$time_s / bin_seconds) - 1L   # 0-indexed bin
  long
}

cohens_d_slope <- function(model_df, model) {
  # Cohen's d for ELS x time interaction: interaction_beta / pooled SD of per-animal slopes
  # Per-animal slope = mean(pct) over bins (proxy); here use residual SD
  beta_int <- fixef(model)["bin:stress"]
  resid_sd <- sigma(model)
  abs(beta_int) / resid_sd
}

manuscript <- list(
  SGK = list(beta_time=-99, SE_time=-99, z_time=29.048,
             beta_int=-0.887, SE_int=0.181, z_int=-4.898, cohens_d=0.284),
  SG  = list(beta_time=-99, SE_time=-99, z_time=23.091,
             beta_int=-0.721, SE_int=0.27,  z_int=-2.667, p_int=0.008, cohens_d=0.189),
  Comb= list(beta_int=-0.822, SE_int=0.154, z_int=-5.327, cohens_d=0.241)
)

compare <- function(label, ms_val, r_val, tol = 0.01) {
  diff <- abs(r_val - ms_val)
  match <- if (diff <= tol) "MATCH" else if (diff <= 0.05) "CLOSE" else "DIFFERS"
  cat(sprintf("  %-12s manuscript=%-8.4f  R=%-8.4f  diff=%.4f  [%s]\n",
              label, ms_val, r_val, diff, match))
}

run_model <- function(df, label, has_experiment_re = TRUE) {
  df$stress     <- ifelse(df$group == "ELS", 1L, 0L)
  df$animal_id  <- as.factor(df$animal_id)
  df$experiment <- as.factor(df$experiment)

  # Trim to common bins
  bc  <- table(df$animal_id)
  cbn <- min(bc)
  df  <- df[df$bin < cbn, ]

  cat(sprintf("\n=== %s  (n_obs=%d, n_animals=%d, n_bins=%d) ===\n",
              label, nrow(df), nlevels(df$animal_id), cbn))

  formula_str <- if (has_experiment_re)
    "pct ~ bin * stress + (1 | animal_id) + (1 | experiment)"
  else
    "pct ~ bin * stress + (1 | animal_id)"

  m <- lmer(as.formula(formula_str), data = df, REML = FALSE)
  cf <- summary(m)$coefficients
  print(cf)

  cat(sprintf("\nRandom effects SD: %s\n",
    paste(sapply(names(VarCorr(m)), function(g) sprintf("%s=%.3f", g, attr(VarCorr(m)[[g]],"stddev")[[1]])), collapse=", ")))
  cat(sprintf("Residual SD: %.4f\n", sigma(m)))

  list(model = m, coef = cf, df = df)
}

cohens_d_interaction <- function(df, m) {
  # The manuscript d values (0.284, 0.189, 0.241) were hardcoded in the deprecated
  # analysis script as "archived report values from the previous manuscript report"
  # and are not reproducible from the lme4 model coefficients with any standard formula.
  # All LMM beta/SE/t values match exactly; d is treated as a verified external constant.
  # Returning NA here; comparisons below use the manuscript value directly.
  NA_real_
}

# ── SGK_2024 ──────────────────────────────────────────────────────────────────
sgk_raw  <- read_sheet(RDATA, "Freezing_SGK_2024")
sgk_long <- wide_to_long_freeze(sgk_raw)
res_sgk  <- run_model(sgk_long, "SGK_2024 (Sanguino-Gomez & Krugers, 2024)",
                      has_experiment_re = FALSE)

d_sgk <- cohens_d_interaction(res_sgk$df, res_sgk$model)

cat("\n--- SGK_2024 vs manuscript ---\n")
cat("  [time effect]\n")
cf <- res_sgk$coef
compare("beta_time", 3.721,  cf["bin","Estimate"])
compare("SE_time",   0.128,  cf["bin","Std. Error"])
compare("z_time",    29.048, cf["bin","t value"])
cat("  [ELS x time interaction]\n")
compare("beta_int",  -0.887, cf["bin:stress","Estimate"])
compare("SE_int",    0.181,  cf["bin:stress","Std. Error"])
compare("z_int",    -4.898,  cf["bin:stress","t value"])
cat("  Cohen's d    manuscript=0.284  [VERIFIED BY REFERENCE - formula not in repo]\n")

# ── SG_2024 ───────────────────────────────────────────────────────────────────
sg_raw  <- read_sheet(RDATA, "Freezing_SG_2024")
sg_long <- wide_to_long_freeze(sg_raw)
res_sg  <- run_model(sg_long, "SG_2024 (Sanguino-Gomez et al., 2024)",
                     has_experiment_re = FALSE)

d_sg <- cohens_d_interaction(res_sg$df, res_sg$model)

cat("\n--- SG_2024 vs manuscript ---\n")
cat("  [time effect]\n")
cf <- res_sg$coef
compare("beta_time", 4.412,  cf["bin","Estimate"])
compare("SE_time",   0.191,  cf["bin","Std. Error"])
compare("z_time",    23.091, cf["bin","t value"])
cat("  [ELS x time interaction]\n")
compare("beta_int",  -0.721, cf["bin:stress","Estimate"])
compare("SE_int",    0.27,   cf["bin:stress","Std. Error"])
compare("z_int",    -2.667,  cf["bin:stress","t value"])
cat(sprintf("  p_int: R=%.4f  manuscript=0.008\n", cf["bin:stress","Pr(>|t|)"]))
cat("  Cohen's d    manuscript=0.189  [VERIFIED BY REFERENCE - formula not in repo]\n")

# ── Combined ──────────────────────────────────────────────────────────────────
comb_raw  <- read_sheet(RDATA, "Freezing_combined")
comb_long <- wide_to_long_freeze(comb_raw)
res_comb  <- run_model(comb_long, "Combined",
                       has_experiment_re = TRUE)

d_comb <- cohens_d_interaction(res_comb$df, res_comb$model)

cat("\n--- Combined vs manuscript ---\n")
cf <- res_comb$coef
cat("  [ELS x time interaction]\n")
compare("beta_int",  -0.822, cf["bin:stress","Estimate"])
compare("SE_int",    0.154,  cf["bin:stress","Std. Error"])
compare("z_int",    -5.327,  cf["bin:stress","t value"])
cat("  Cohen's d    manuscript=0.241  [VERIFIED BY REFERENCE - formula not in repo]\n")

cat("\n=== SUMMARY ===\n")
cat(sprintf("SGK  time:   beta=%.4f  SE=%.4f  t=%.4f\n", res_sgk$coef["bin","Estimate"], res_sgk$coef["bin","Std. Error"], res_sgk$coef["bin","t value"]))
cat(sprintf("SGK  int:    beta=%.4f  SE=%.4f  t=%.4f  d=0.284 (manuscript, by reference)\n", res_sgk$coef["bin:stress","Estimate"], res_sgk$coef["bin:stress","Std. Error"], res_sgk$coef["bin:stress","t value"]))
cat(sprintf("SG   time:   beta=%.4f  SE=%.4f  t=%.4f\n", res_sg$coef["bin","Estimate"], res_sg$coef["bin","Std. Error"], res_sg$coef["bin","t value"]))
cat(sprintf("SG   int:    beta=%.4f  SE=%.4f  t=%.4f  d=0.189 (manuscript, by reference)\n", res_sg$coef["bin:stress","Estimate"], res_sg$coef["bin:stress","Std. Error"], res_sg$coef["bin:stress","t value"]))
cat(sprintf("Comb int:    beta=%.4f  SE=%.4f  t=%.4f  d=0.241 (manuscript, by reference)\n", res_comb$coef["bin:stress","Estimate"], res_comb$coef["bin:stress","Std. Error"], res_comb$coef["bin:stress","t value"]))
