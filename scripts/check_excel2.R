suppressPackageStartupMessages(library(readxl))

f1 <- "C:/Users/jenif/Downloads/Raw_data (1).xlsx"
f2 <- "C:/Users/jenif/Downloads/Statistical_report.xlsx"

cat("=== Raw_data: Cluster_frequency ===\n")
cf <- read_excel(f1, sheet = "Cluster_frequency")
print(cf, n = 20)

cat("\n=== Raw_data: Supp_cluster_frequency ===\n")
scf <- read_excel(f1, sheet = "Supp_cluster_frequency")
print(scf, n = 20)

cat("\n=== Statistical_report: Fig.4A_Clusters_frequency ===\n")
sr_freq <- read_excel(f2, sheet = "Fig.4A_Clusters_frequency")
print(sr_freq, n = 40)
