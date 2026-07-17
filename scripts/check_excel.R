suppressPackageStartupMessages(library(readxl))

f1 <- "C:/Users/jenif/Downloads/Raw_data (1).xlsx"
f2 <- "C:/Users/jenif/Downloads/Statistical_report.xlsx"

cat("=== Raw_data (1).xlsx sheets ===\n")
print(excel_sheets(f1))

cat("\n=== Statistical_report.xlsx sheets ===\n")
print(excel_sheets(f2))
