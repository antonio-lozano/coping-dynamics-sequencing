suppressPackageStartupMessages({
  library(lme4); library(lmerTest); library(geepack); library(glmmTMB); library(dplyr); library(readxl)
})

FREQ_CSV  <- "D:/coping-dynamics-sequencing/data/source/cluster_frequency_per_animal.csv"
TIME_CSV  <- "D:/coping-dynamics-sequencing/data/source/cluster_timecourse_per_animal.csv"
TRACK_CSV <- "D:/coping-dynamics-sequencing/data/source/tracking_exclusions_per_animal.csv"

compare <- function(label, ms_val, r_val, tol = 0.01) {
  diff <- abs(r_val - ms_val)
  tag  <- if (diff <= tol) "MATCH" else if (diff <= 0.05) "CLOSE" else "DIFFERS"
  cat(sprintf("  %-12s  ms=%-9.4f  R=%-9.4f  diff=%.4f  [%s]\n", label, ms_val, r_val, diff, tag))
}

prep <- function(df) {
  df$stress     <- ifelse(df$group == "ELS", 1L, 0L)
  df$animal_id  <- as.factor(df$animal_id)
  df$experiment <- as.factor(as.character(df$experiment))
  df
}

# ═══════════════════════════════════════════════════════════════════════════════
# 1. TRACKING EXCLUSIONS (Supp Fig 1)
#    β = −2.341, SE = 0.942, z = −2.486, p = 0.013
#    "slightly higher tracking exclusions in controls"
# ═══════════════════════════════════════════════════════════════════════════════
cat("\n════ TRACKING EXCLUSIONS (Supp Fig 1) ════\n")
excl <- prep(read.csv(TRACK_CSV))

# LMM on inaccurate tracking percentage (inaccurate_tracking_seconds / 450 * 100)
excl$pct_excluded <- excl$inaccurate_tracking_seconds / 450 * 100

cat("\n--- LMM: inaccurate tracking % (inaccurate_seconds/450*100) ---\n")
m_excl <- lmer(pct_excluded ~ stress + (1|experiment), data = excl, REML = FALSE)
cf <- summary(m_excl)$coefficients
print(cf)
cat(sprintf("RE: experiment SD=%.3f  Residual=%.3f\n", attr(VarCorr(m_excl)[["experiment"]],"stddev")[[1]], sigma(m_excl)))
compare("beta",  -2.341, cf["stress","Estimate"])
compare("SE",     0.942, cf["stress","Std. Error"])
compare("z/t",   -2.486, cf["stress","t value"])
compare("p",      0.013, cf["stress","Pr(>|t|)"])

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CLUSTER FREQUENCY Fig 3A (GEE NB / GLMM NB, log link)
#    Freeze:     β = −0.204, SE = 0.081, z = −2.510, p = 0.012
#    Turn:       β =  0.101, SE = 0.032, z =  3.099, p = 0.002
#    Sniff:      β =  0.621, SE = 0.231, z =  2.694, p = 0.007
#    Locomotion: β = −0.158, SE = 0.161, z = −0.978, p = 0.328
# ═══════════════════════════════════════════════════════════════════════════════
cat("\n\n════ CLUSTER FREQUENCY Fig 3A (GEE NB, log link) ════\n")
freq_df <- prep(read.csv(FREQ_CSV))

ms_freq <- list(
  Freeze     = list(beta = -0.204, SE = 0.081, z = -2.510, p = 0.012),
  Turn       = list(beta =  0.101, SE = 0.032, z =  3.099, p = 0.002),
  Sniff      = list(beta =  0.621, SE = 0.231, z =  2.694, p = 0.007),
  Locomotion = list(beta = -0.158, SE = 0.161, z = -0.978, p = 0.328)
)

for (cl in names(ms_freq)) {
  sub <- freq_df[freq_df$cluster == cl, ]
  sub$freq_frames <- round(sub$frequency_seconds * 25)  # convert to frame counts (25 FPS)
  sub$exp_id <- as.integer(sub$experiment)

  cat(sprintf("\n=== %s (n=%d) ===\n", cl, nrow(sub)))
  cat(sprintf("  Control mean=%.2fs  ELS mean=%.2fs  log(ELS/Ctrl)=%.4f\n",
              mean(sub$frequency_seconds[sub$stress==0]),
              mean(sub$frequency_seconds[sub$stress==1]),
              log(mean(sub$frequency_seconds[sub$stress==1])/mean(sub$frequency_seconds[sub$stress==0]))))

  # GEE NB
  tryCatch({
    m_gee <- geeglm(freq_frames ~ stress, id = exp_id, data = sub,
                    family = negative.binomial(1), corstr = "exchangeable")
    cf <- summary(m_gee)$coefficients
    cat("GEE NB:\n"); print(cf)
    compare("beta",  ms_freq[[cl]]$beta, cf["stress","Estimate"])
    compare("SE",    ms_freq[[cl]]$SE,   cf["stress","Std.err"])
    compare("z",     ms_freq[[cl]]$z,    cf["stress","Wald"])
    compare("p",     ms_freq[[cl]]$p,    cf["stress","Pr(>|W|)"])
  }, error = function(e) cat("GEE NB failed:", conditionMessage(e), "\n"))

  # Also try GLMM NB (glmmTMB) with experiment RE
  tryCatch({
    m_glmm <- glmmTMB(freq_frames ~ stress + (1|experiment), data = sub,
                      family = nbinom2)
    cf2 <- summary(m_glmm)$coefficients$cond
    cat("GLMM NB (glmmTMB, experiment RE):\n"); print(cf2)
    compare("beta",  ms_freq[[cl]]$beta, cf2["stress","Estimate"])
    compare("SE",    ms_freq[[cl]]$SE,   cf2["stress","Std. Error"])
    compare("z",     ms_freq[[cl]]$z,    cf2["stress","z value"])
    compare("p",     ms_freq[[cl]]$p,    cf2["stress","Pr(>|z|)"])
  }, error = function(e) cat("GLMM NB failed:", conditionMessage(e), "\n"))
}

# ═══════════════════════════════════════════════════════════════════════════════
# 3. CLUSTER TIMECOURSE Fig 3B-H (LMM, ELS×time interaction)
#    Freeze:     β = −0.023, SE = 0.005, z = −4.241
#    Sniff:      β = −0.014, SE = 0.004, z = −3.656
#    Turn:       β =  0.026, SE = 0.007, z =  3.610
#    Locomotion: β = −1.859, SE = 0.834, z = −2.229
# ═══════════════════════════════════════════════════════════════════════════════
cat("\n\n════ CLUSTER TIMECOURSE Fig 3B-H (LMM, interaction) ════\n")
tc_df <- prep(read.csv(TIME_CSV))

ms_time <- list(
  Freeze     = list(beta = -0.023, SE = 0.005, z = -4.241),
  Sniff      = list(beta = -0.014, SE = 0.004, z = -3.656),
  Turn       = list(beta =  0.026, SE = 0.007, z =  3.610),
  Locomotion = list(beta = -1.859, SE = 0.834, z = -2.229)
)

for (cl in names(ms_time)) {
  sub <- tc_df[tc_df$cluster == cl, ]
  sub$stress     <- ifelse(sub$group == "ELS", 1L, 0L)
  sub$animal_id  <- as.factor(sub$animal_id)
  sub$experiment <- as.factor(as.character(sub$experiment))
  sub <- sub[!is.na(sub$pct), ]

  # Trim to minimum bin count across animals
  bc  <- table(sub$animal_id)
  cbn <- min(bc)
  keep_t <- sort(unique(sub$time_s))[1:cbn]
  sub <- sub[sub$time_s %in% keep_t, ]

  cat(sprintf("\n=== %s timecourse (n_obs=%d, n_animals=%d, n_bins=%d) ===\n",
              cl, nrow(sub), nlevels(sub$animal_id), cbn))

  tryCatch({
    m <- lmer(pct ~ time_s * stress + (1|animal_id) + (1|experiment),
              data = sub, REML = FALSE)
    cf <- summary(m)$coefficients
    print(cf)
    cat(sprintf("RE SD: animal=%.3f  experiment=%.3f  Residual=%.3f\n",
                attr(VarCorr(m)[["animal_id"]],"stddev")[[1]],
                attr(VarCorr(m)[["experiment"]],"stddev")[[1]],
                sigma(m)))
    cat(sprintf("\n--- %s vs manuscript ---\n", cl))
    compare("beta_int",  ms_time[[cl]]$beta, cf["time_s:stress","Estimate"])
    compare("SE_int",    ms_time[[cl]]$SE,   cf["time_s:stress","Std. Error"])
    compare("z_int",     ms_time[[cl]]$z,    cf["time_s:stress","t value"])
  }, error = function(e) cat(cl, "failed:", conditionMessage(e), "\n"))
}

cat("\n\n════ DONE ════\n")
