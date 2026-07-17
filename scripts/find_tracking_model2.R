suppressPackageStartupMessages(library(readxl))

f2 <- "C:/Users/jenif/Downloads/Statistical_report.xlsx"

cat("=== Fig.3H_Freezing_syllables ===\n")
d <- read_excel(f2, sheet = "Fig.3H_Freezing_syllables", col_names = FALSE)
cat("Dims:", nrow(d), "x", ncol(d), "\n")
print(d[1:15, 1:8])

cat("\n=== Fig.3A-C_Ground_truth (full) ===\n")
d2 <- read_excel(f2, sheet = "Fig.3A-C_Ground_truth ", col_names = FALSE)
cat("Dims:", nrow(d2), "x", ncol(d2), "\n")
# Search for tracking/inaccurate
for (i in 1:nrow(d2)) {
  row_txt <- paste(as.character(unlist(d2[i,])), collapse=" ")
  if (grepl("2\\.341|2\\.486|track|inaccur|Inaccur|exclus|mix|Mix", row_txt, ignore.case=TRUE)) {
    cat(sprintf("Row %d: %s\n", i, row_txt))
  }
}
