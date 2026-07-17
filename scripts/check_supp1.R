suppressPackageStartupMessages({ library(lme4); library(lmerTest) })

excl <- read.csv("D:/coping-dynamics-sequencing/data/source/tracking_exclusions_per_animal.csv")
excl$stress     <- ifelse(excl$group == "ELS", 1L, 0L)
excl$experiment <- as.factor(as.character(excl$experiment))

TRIAL_S <- 450

# ── ~20% claim ────────────────────────────────────────────────────
excl$pct_total    <- excl$total_excluded_seconds    / TRIAL_S * 100
excl$pct_inaccurate <- excl$inaccurate_tracking_seconds / TRIAL_S * 100
excl$pct_mix      <- excl$mix_behaviors_seconds     / TRIAL_S * 100

cat("=== Overall exclusion percentages ===\n")
cat(sprintf("Total excluded:      Control=%.1f%%  ELS=%.1f%%  Overall=%.1f%%\n",
    mean(excl$pct_total[excl$stress==0]),
    mean(excl$pct_total[excl$stress==1]),
    mean(excl$pct_total)))
cat(sprintf("Inaccurate tracking: Control=%.1f%%  ELS=%.1f%%  Overall=%.1f%%\n",
    mean(excl$pct_inaccurate[excl$stress==0]),
    mean(excl$pct_inaccurate[excl$stress==1]),
    mean(excl$pct_inaccurate)))
cat(sprintf("Mix behaviors:       Control=%.1f%%  ELS=%.1f%%  Overall=%.1f%%\n",
    mean(excl$pct_mix[excl$stress==0]),
    mean(excl$pct_mix[excl$stress==1]),
    mean(excl$pct_mix)))

# ── Inaccurate tracking LMM (manuscript stat) ─────────────────────
cat("\n=== LMM: inaccurate tracking % ===\n")
m1 <- lmer(pct_inaccurate ~ stress + (1|experiment), data=excl, REML=FALSE)
cf1 <- summary(m1)$coefficients
print(round(cf1, 4))
cat(sprintf("Manuscript: beta=-2.341  SE=0.942  z=-2.486  p=0.013\n"))
cat(sprintf("Our model:  beta=%.3f  SE=%.3f  z=%.3f  p=%.3f\n",
    cf1["stress","Estimate"], cf1["stress","Std. Error"],
    cf1["stress","t value"], cf1["stress","Pr(>|t|)"]))

# ── Mix behaviors LMM (should show NO difference) ─────────────────
cat("\n=== LMM: mix behaviors % ===\n")
m2 <- lmer(pct_mix ~ stress + (1|experiment), data=excl, REML=FALSE)
cf2 <- summary(m2)$coefficients
print(round(cf2, 4))
cat(sprintf("Mix stress effect: beta=%.3f  SE=%.3f  z=%.3f  p=%.3f\n",
    cf2["stress","Estimate"], cf2["stress","Std. Error"],
    cf2["stress","t value"], cf2["stress","Pr(>|t|)"]))

# ── Total exclusions LMM ──────────────────────────────────────────
cat("\n=== LMM: total exclusions % ===\n")
m3 <- lmer(pct_total ~ stress + (1|experiment), data=excl, REML=FALSE)
cf3 <- summary(m3)$coefficients
print(round(cf3, 4))
cat(sprintf("Total stress effect: beta=%.3f  SE=%.3f  z=%.3f  p=%.3f\n",
    cf3["stress","Estimate"], cf3["stress","Std. Error"],
    cf3["stress","t value"], cf3["stress","Pr(>|t|)"]))
