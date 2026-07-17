suppressPackageStartupMessages(library(readxl))

f1 <- "C:/Users/jenif/Downloads/Raw_data (1).xlsx"
f2 <- "C:/Users/jenif/Downloads/Statistical_report.xlsx"

# Check all Statistical_report sheets for tracking exclusion stats
cat("=== Searching Statistical_report for tracking exclusion model ===\n")
for (sh in excel_sheets(f2)) {
  d <- read_excel(f2, sheet = sh, col_names = FALSE)
  # Look for -2.341 or "track" or "exclusion" or "inaccurate"
  txt <- apply(d, 2, as.character)
  if (any(grepl("2\\.341|2\\.486|track|Inaccur|exclus", txt, ignore.case=TRUE))) {
    cat(sprintf("\n  FOUND in sheet: '%s'\n", sh))
    print(d[1:10, 1:10])
  }
}

# Check Supp sheets in Raw_data
cat("\n=== Raw_data: Supp_cluster_frequency ===\n")
scf <- read_excel(f1, sheet = "Supp_cluster_frequency", col_names = FALSE)
cat("Dims:", nrow(scf), "x", ncol(scf), "\n")
print(scf[1:6, 1:6])

cat("\n=== Raw_data: Supp_cluster_time ===\n")
sct <- read_excel(f1, sheet = "Supp_cluster_time", col_names = FALSE)
cat("Dims:", nrow(sct), "x", ncol(sct), "\n")
print(sct[1:6, 1:8])
