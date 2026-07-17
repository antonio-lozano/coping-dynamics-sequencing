suppressPackageStartupMessages(library(readxl))

f2 <- "C:/Users/jenif/Downloads/Statistical_report.xlsx"
tc <- read_excel(f2, sheet = "Fig.4B-I_Clusters_over_time", col_names = FALSE)

# Cluster positions from row 1:
# Freeze=1, Jump=19, Locomotion=38, Climb=56, Turn=74, Sniff=92, Groom=110,
# Mix behaviors=128, Inaccurate tracking=146, Not mapped=164
r1 <- as.character(unlist(tc[1, ]))
cat("Cluster positions:\n")
pos <- which(!is.na(r1))
print(data.frame(cluster=r1[pos], col=pos))

# Read Inaccurate tracking section (col 146)
cat("\n=== Inaccurate tracking (col 146) — rows 1-10 ===\n")
inac_col <- 146
print(tc[1:10, inac_col:(inac_col+9)])

cat("\n=== Mix behaviors (col 128) — rows 1-10 ===\n")
mix_col <- 128
print(tc[1:10, mix_col:(mix_col+9)])
