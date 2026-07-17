suppressPackageStartupMessages({ library(geepack); library(MASS) })

freq_df <- read.csv("D:/coping-dynamics-sequencing/data/source/cluster_frequency_per_animal.csv")
freq_df$stress     <- ifelse(freq_df$group == "ELS", 1L, 0L)
freq_df$experiment <- as.integer(as.factor(as.character(freq_df$experiment)))
freq_df$freq_frames <- round(freq_df$frequency_seconds * 25)

ms <- list(
  Freeze     = c(beta=-0.204, SE=0.081, z=-2.510, p=0.012),
  Sniff      = c(beta=0.621,  SE=0.231, z=2.694,  p=0.007),
  Turn       = c(beta=0.101,  SE=0.032, z=3.099,  p=0.002)
)

compare <- function(label, ms_val, gee_val, tol=0.001) {
  diff <- abs(gee_val - ms_val)
  tag  <- if (diff <= tol) "MATCH" else if (diff <= 0.05) "CLOSE" else "DIFFERS"
  cat(sprintf("    %-6s  paragraph=%-8.4f  GEE=%-8.4f  diff=%.4f  [%s]\n",
      label, ms_val, gee_val, diff, tag))
}

for (cl in names(ms)) {
  sub <- freq_df[freq_df$cluster == cl, ]
  sub <- sub[order(sub$experiment, sub$animal_id), ]

  cat(sprintf("\n=== %s (n=%d) ===\n", cl, nrow(sub)))

  # GEE NB with experiment as clustering variable (population-averaged)
  m <- geeglm(freq_frames ~ stress + experiment,
              id       = experiment,
              data     = sub,
              family   = negative.binomial(1),
              corstr   = "exchangeable")

  cf <- summary(m)$coefficients
  cat("GEE NB coefficients:\n")
  print(round(cf, 6))

  compare("beta", ms[[cl]]["beta"], cf["stress", "Estimate"])
  compare("SE",   ms[[cl]]["SE"],   cf["stress", "Std.err"])
  compare("z",    ms[[cl]]["z"],    cf["stress", "Wald"])
  compare("p",    ms[[cl]]["p"],    cf["stress", "Pr(>|W|)"])
}
