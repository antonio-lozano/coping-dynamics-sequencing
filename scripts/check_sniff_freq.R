suppressPackageStartupMessages({ library(glmmTMB); library(lme4) })

freq_df <- read.csv("D:/coping-dynamics-sequencing/data/source/cluster_frequency_per_animal.csv")
freq_df$stress     <- ifelse(freq_df$group == "ELS", 1L, 0L)
freq_df$experiment <- as.factor(as.character(freq_df$experiment))

sub <- freq_df[freq_df$cluster == "Sniff", ]
sub$freq_frames <- round(sub$frequency_seconds * 25)

cat(sprintf("n=%d  Control mean=%.3fs  ELS mean=%.3fs\n",
    nrow(sub),
    mean(sub$frequency_seconds[sub$stress==0]),
    mean(sub$frequency_seconds[sub$stress==1])))
cat(sprintf("log ratio (ELS/Ctrl) = %.4f\n",
    log(mean(sub$frequency_seconds[sub$stress==1]) / mean(sub$frequency_seconds[sub$stress==0]))))

cat("\n=== GLMM NB (nbinom2, experiment RE, freq_frames) ===\n")
m1 <- glmmTMB(freq_frames ~ stress + (1|experiment), data = sub, family = nbinom2)
cf1 <- summary(m1)$coefficients$cond
print(round(cf1, 4))

cat("\n=== GLMM NB (nbinom2, no RE, freq_frames) ===\n")
m3 <- glmmTMB(freq_frames ~ stress, data = sub, family = nbinom2)
cf3 <- summary(m3)$coefficients$cond
print(round(cf3, 4))

cat("\n=== GLMM Poisson (experiment RE, freq_frames) ===\n")
m5 <- glmmTMB(freq_frames ~ stress + (1|experiment), data = sub, family = poisson)
cf5 <- summary(m5)$coefficients$cond
print(round(cf5, 4))

cat("\n=== LMM (REML=FALSE, experiment RE, frequency_seconds) ===\n")
m6 <- lmer(frequency_seconds ~ stress + (1|experiment), data = sub, REML = FALSE)
cf6 <- summary(m6)$coefficients
print(round(cf6, 4))

cat("\nManuscript paragraph: beta=+0.621, SE=0.231, z=+2.694, p=0.007\n")
cat("REPORTED list:        beta=-0.068, SE=0.185, z=-2.133, p=0.033  [FLAGGED z!=beta/SE]\n")
cat("True beta/SE from REPORTED: -0.068/0.185 =", round(-0.068/0.185, 3), "\n")
