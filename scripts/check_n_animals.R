suppressPackageStartupMessages(library(readxl))

f1 <- "C:/Users/jenif/Downloads/Raw_data (1).xlsx"
f2 <- "C:/Users/jenif/Downloads/Statistical_report.xlsx"

# Fig 1A SimBA sheet
cat("=== Statistical_report: Fig.1A_SimBA_validation ===\n")
s1 <- read_excel(f2, sheet = "Fig.1A_ SimBA_validation", col_names = FALSE)
print(s1)

# SimBA_validation raw data
cat("\n=== Raw_data: SimBA_validation (first rows) ===\n")
sv <- read_excel(f1, sheet = "SimBA_validation")
cat("Columns:", names(sv), "\n")
cat("n rows:", nrow(sv), "\n")
print(head(sv, 3))

# Freezing sheets
for (sh in c("Freezing_SGK_2024", "Freezing_SG_2024", "Freezing_combined")) {
  cat(sprintf("\n=== Raw_data: %s ===\n", sh))
  d <- read_excel(f1, sheet = sh, col_names = FALSE)
  # Row 2 should be headers
  print(d[1:4, 1:6])
  cat("Total rows:", nrow(d), "\n")
}
