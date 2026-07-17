suppressPackageStartupMessages(library(readxl))

f2 <- "C:/Users/jenif/Downloads/Statistical_report.xlsx"

compare <- function(label, ms_val, sr_val, tol = 0.001) {
  diff <- abs(sr_val - ms_val)
  tag  <- if (diff <= tol) "MATCH" else if (diff <= 0.05) "CLOSE" else "DIFFERS"
  cat(sprintf("  %-30s  paragraph=%-8.4f  report=%-8.4f  diff=%.4f  [%s]\n",
      label, ms_val, sr_val, diff, tag))
}

# ─────────────────────────────────────────────────────────────────
# 1. CLUSTER FREQUENCY (Fig 3A) — Statistical_report: Fig.4A
# ─────────────────────────────────────────────────────────────────
cat("════ Fig 3A — Cluster Frequency (GEE NB) ════\n")
sr <- read_excel(f2, sheet = "Fig.4A_Clusters_frequency", col_names = FALSE)

# Column positions (row 1): Freeze=2, Jump=15, Locomotion=28, Climb=41, Turn=54, Sniff=67, Groom=80
cluster_cols <- c(Freeze=2, Locomotion=28, Turn=54, Sniff=67)
ms_freq <- list(
  Freeze     = c(beta=-0.204, SE=0.081, z=-2.510, p=0.012),
  Locomotion = c(beta=-0.158, SE=0.161, z=-0.978, p=0.328),
  Turn       = c(beta=0.101,  SE=0.032, z=3.099,  p=0.002),
  Sniff      = c(beta=0.621,  SE=0.231, z=2.694,  p=0.007)
)

for (cl in names(cluster_cols)) {
  col <- cluster_cols[[cl]]
  row <- as.numeric(as.character(unlist(sr[4, col:(col+6)])))
  cat(sprintf("\n  %s:\n", cl))
  compare(paste(cl, "beta"), ms_freq[[cl]]["beta"], row[2])
  compare(paste(cl, "SE"),   ms_freq[[cl]]["SE"],   row[3])
  compare(paste(cl, "z"),    ms_freq[[cl]]["z"],     row[4])
  compare(paste(cl, "p"),    ms_freq[[cl]]["p"],     row[5])
}

# ─────────────────────────────────────────────────────────────────
# 2. CLUSTER TIMECOURSE (Fig 3B-H) — Statistical_report: Fig.4B-I
# ─────────────────────────────────────────────────────────────────
cat("\n\n════ Fig 3B-H — Cluster Timecourse (LMM, stress×time interaction) ════\n")
tc <- read_excel(f2, sheet = "Fig.4B-I_Clusters_over_time", col_names = FALSE)

# Row 1 cluster positions: Freeze=1, Jump=19, Locomotion=38, Climb=56, Turn=74, Sniff=92, Groom=110
cluster_tc_cols <- c(Freeze=1, Locomotion=38, Turn=74, Sniff=92)

ms_time <- list(
  Freeze     = list(fig="3B", stress=c(beta=NA,    SE=NA,    z=NA),
                               inter=c(beta=-0.023, SE=0.005, z=-4.241, p=0)),
  Sniff      = list(fig="3C", stress=c(beta=NA,    SE=NA,    z=NA),
                               inter=c(beta=-0.014, SE=0.004, z=-3.656, p=0)),
  Turn       = list(fig="3E", stress=c(beta=NA,    SE=NA,    z=NA),
                               inter=c(beta=0.026,  SE=0.007, z=3.610,  p=0)),
  Locomotion = list(fig="3F", stress=c(beta=-1.859, SE=0.834, z=-2.229, p=0.026),
                               inter=c(beta=0.005,  SE=0.002, z=2.553,  p=0.011))
)

for (cl in names(cluster_tc_cols)) {
  col <- cluster_tc_cols[[cl]]
  # Row 4 = Stress (main effect), Row 6 = Stress x time (interaction)
  stress_row <- as.numeric(as.character(unlist(tc[4, col:(col+7)])))
  inter_row  <- as.numeric(as.character(unlist(tc[6, col:(col+7)])))

  cat(sprintf("\n  %s (Fig %s):\n", cl, ms_time[[cl]]$fig))
  cat("    --- Stress main effect ---\n")
  cat(sprintf("    report: beta=%.4f  SE=%.4f  z=%.4f  p=%.6f\n",
      stress_row[2], stress_row[3], stress_row[4], stress_row[5]))

  cat("    --- Stress x time interaction ---\n")
  cat(sprintf("    report: beta=%.4f  SE=%.4f  z=%.4f  p=%.6f\n",
      inter_row[2], inter_row[3], inter_row[4], inter_row[5]))

  if (!is.na(ms_time[[cl]]$inter["beta"])) {
    compare(paste(cl, "inter beta"), ms_time[[cl]]$inter["beta"], inter_row[2])
    compare(paste(cl, "inter SE"),   ms_time[[cl]]$inter["SE"],   inter_row[3])
    compare(paste(cl, "inter z"),    ms_time[[cl]]$inter["z"],    inter_row[4])
  }
  if (!is.na(ms_time[[cl]]$stress["beta"])) {
    compare(paste(cl, "stress beta"), ms_time[[cl]]$stress["beta"], stress_row[2])
    compare(paste(cl, "stress SE"),   ms_time[[cl]]$stress["SE"],   stress_row[3])
    compare(paste(cl, "stress z"),    ms_time[[cl]]$stress["z"],    stress_row[4])
    compare(paste(cl, "stress p"),    ms_time[[cl]]$stress["p"],    stress_row[5])
  }
}

cat("\n\n════ DONE ════\n")
