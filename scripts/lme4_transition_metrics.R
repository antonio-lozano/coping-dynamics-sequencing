library(lmerTest)

data <- read.csv("d:/coping-dynamics-sequencing/results/figure_data/transition_metrics_per_animal.csv")
data$Condition  <- factor(data$Condition, levels = c("Control", "ELS"))
data$Experiment <- factor(data$Experiment)

cat("N animals:", nrow(data), "\n")
cat("Condition:", table(data$Condition), "\n")
cat("Experiment:", table(data$Experiment), "\n\n")

gold <- list(
  LZ          = list(b=-8.04878,  se=5.56673, z=-1.446, p=0.148),
  Recurrence  = list(b= 0.01344,  se=0.00594, z= 2.264, p=0.024),
  Determinism = list(b= 0.01610,  se=0.00476, z= 3.382, p=0.001),
  Markov      = list(b=-0.04119,  se=0.01759, z=-2.342, p=0.019)
)

run_lm <- function(formula_str, data, param, metric_name, g) {
  cat(sprintf("─── %s ───\n", metric_name))
  m <- lm(as.formula(formula_str), data = data)
  s <- summary(m)$coefficients
  b  <- s[param, "Estimate"]
  se <- s[param, "Std. Error"]
  t  <- s[param, "t value"]
  p  <- s[param, "Pr(>|t|)"]
  df <- m$df.residual
  # z approximation (normal) — comparable to statsmodels output
  z_approx <- b / se
  p_z <- 2 * pnorm(-abs(z_approx))

  cat(sprintf("  lm:        beta=%.5f  SE=%.5f  t(df=%d)=%.3f  p=%.5f\n", b, se, df, t, p))
  cat(sprintf("  z-approx:  z=%.3f  p_z=%.5f\n", z_approx, p_z))
  cat(sprintf("  Gold:      beta=%.5f  SE=%.5f  z=%.3f  p=%.5f\n", g$b, g$se, g$z, g$p))
  cat(sprintf("  SE diff: %.5f  |  sig (t-based): %s  |  sig (z-approx): %s\n\n",
              abs(se - g$se),
              ifelse(p < 0.05, "YES", "NO"),
              ifelse(p_z < 0.05, "YES", "NO")))
}

run_lm("LZ_complexity   ~ Condition + Experiment", data, "ConditionELS", "LZ",          gold$LZ)
run_lm("Recurrence_Rate ~ Condition + Experiment", data, "ConditionELS", "Recurrence",  gold$Recurrence)
run_lm("Determinism     ~ Condition + Experiment", data, "ConditionELS", "Determinism", gold$Determinism)
run_lm("MarkovEntropy   ~ Condition + Experiment", data, "ConditionELS", "Markov",      gold$Markov)

cat("=== SUMMARY TABLE ===\n")
cat(sprintf("%-12s  %8s  %8s  %8s  %8s  %6s\n", "Metric", "beta", "SE", "t", "p(t)", "sig?"))
for (item in list(
  list("LZ",          "LZ_complexity",   gold$LZ),
  list("Recurrence",  "Recurrence_Rate", gold$Recurrence),
  list("Determinism", "Determinism",     gold$Determinism),
  list("Markov",      "MarkovEntropy",   gold$Markov)
)) {
  m <- lm(paste(item[[2]], "~ Condition + Experiment"), data = data)
  s <- summary(m)$coefficients["ConditionELS",]
  cat(sprintf("%-12s  %8.5f  %8.5f  %8.3f  %8.5f  %6s\n",
              item[[1]], s["Estimate"], s["Std. Error"], s["t value"], s["Pr(>|t|)"],
              ifelse(s["Pr(>|t|)"] < 0.05, "YES", "NO")))
}
