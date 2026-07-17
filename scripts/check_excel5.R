suppressPackageStartupMessages(library(readxl))

f2 <- "C:/Users/jenif/Downloads/Statistical_report.xlsx"
tc <- read_excel(f2, sheet = "Fig.4B-I_Clusters_over_time", col_names = FALSE)

# Clusters start at col 1, each takes 17 cols based on row 2 pattern
# Row 1: cluster names at cols 1, 18 (?), let's find them
r1 <- as.character(unlist(tc[1, ]))
cluster_positions <- which(!is.na(r1))
cat("Cluster header positions:", cluster_positions, "\n")
cat("Cluster names:", r1[cluster_positions], "\n")

# Print rows 3-10 for Freeze (cols 1-10) and Locomotion (cols ~21-30)
cat("\n=== Freeze (cols 1-10) rows 3-10 ===\n")
print(tc[3:10, 1:10])

cat("\n=== Locomotion rows 3-10 ===\n")
loco_start <- cluster_positions[3]
print(tc[3:10, loco_start:(loco_start+9)])

cat("\n=== Turn rows 3-10 ===\n")
turn_start <- cluster_positions[5]
print(tc[3:10, turn_start:(turn_start+9)])

cat("\n=== Sniff rows 3-10 ===\n")
sniff_start <- cluster_positions[6]
print(tc[3:10, sniff_start:(sniff_start+9)])
