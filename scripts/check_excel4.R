suppressPackageStartupMessages(library(readxl))

f2 <- "C:/Users/jenif/Downloads/Statistical_report.xlsx"

sr <- read_excel(f2, sheet = "Fig.4A_Clusters_frequency", col_names = FALSE)

# Cluster column start positions from row 1
# Freeze=2, Jump=15, Locomotion=28, Climb=41, Turn=54, Sniff=67, Groom=80
cluster_cols <- c(Freeze=2, Jump=15, Locomotion=28, Climb=41, Turn=54, Sniff=67, Groom=80)

cat("=== Combined dataset — cluster frequency (GEE) ===\n")
cat(sprintf("%-12s  %8s  %8s  %8s  %8s\n", "Cluster", "beta", "SE", "z", "p"))
cat(strrep("-", 56), "\n")

for (cl in names(cluster_cols)) {
  col <- cluster_cols[[cl]]
  # Row 3 = "Combined" intercept, Row 4 = Stress
  stress_row <- sr[4, col:(col+6)]
  vals <- as.numeric(as.character(unlist(stress_row)))
  cat(sprintf("%-12s  %8.4f  %8.4f  %8.4f  %8.4f\n",
      cl, vals[2], vals[3], vals[4], vals[5]))
}

cat("\n=== Statistical_report: Fig.4B-I_Clusters_over_time ===\n")
tc <- read_excel(f2, sheet = "Fig.4B-I_Clusters_over_time", col_names = FALSE)
cat("Dimensions:", nrow(tc), "x", ncol(tc), "\n")
cat("\nRow 1:\n")
r1 <- as.character(unlist(tc[1, ]))
print(r1[!is.na(r1)])
cat("\nRow 2:\n")
r2 <- as.character(unlist(tc[2, ]))
print(r2[!is.na(r2)])
