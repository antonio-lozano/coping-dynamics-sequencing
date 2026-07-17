suppressPackageStartupMessages({
  library(geepack)
  library(MASS)
})

tc <- read.csv("D:/coping-dynamics-sequencing/data/source/cluster_timecourse_per_animal.csv")
tc$stress <- ifelse(tc$group == "ELS", 1L, 0L)
tc$experiment <- as.factor(as.character(tc$experiment))
exp_vals <- sort(unique(as.character(tc$experiment)))
tc$exp_bin <- ifelse(as.character(tc$experiment) == exp_vals[2], 1L, 0L)
tc$frames_in_bin <- round(tc$pct / 100 * 750)

try_one <- function(label, expr) {
  cat("\n", label, "\n", sep="")
  res <- tryCatch(expr, error=function(e) e)
  if (inherits(res, "error")) {
    cat("FAILED:", conditionMessage(res), "\n")
  } else {
    cf <- summary(res)$coefficients
    print(cf)
    if ("stress" %in% rownames(cf)) {
      cat(sprintf("stress beta=%.4f SE=%.4f z=%.4f p=%.6f\n",
                  cf["stress","Estimate"], cf["stress","Std.err"],
                  cf["stress","Wald"], cf["stress","Pr(>|W|)"]))
    }
  }
}

sub <- tc[tc$cluster == "Freeze", ]
sub <- sub[order(sub$animal_id, sub$time_s), ]
cat(sprintf("Freeze rows=%d animals=%d zeros=%d\n", nrow(sub), length(unique(sub$animal_id)), sum(sub$frames_in_bin == 0)))

try_one("geeglm NB exch id animal", geeglm(frames_in_bin ~ stress + exp_bin,
  id=animal_id, data=sub, family=negative.binomial(theta=1), corstr="exchangeable"))

try_one("geeglm NB indep id animal", geeglm(frames_in_bin ~ stress + exp_bin,
  id=animal_id, data=sub, family=negative.binomial(theta=1), corstr="independence"))

try_one("geeglm Poisson exch id animal", geeglm(frames_in_bin ~ stress + exp_bin,
  id=animal_id, data=sub, family=poisson(link="log"), corstr="exchangeable"))

try_one("geeglm NB exch id experiment", geeglm(frames_in_bin ~ stress + exp_bin,
  id=experiment, data=sub, family=negative.binomial(theta=1), corstr="exchangeable"))

try_one("glm.nb", glm.nb(frames_in_bin ~ stress + exp_bin, data=sub))
