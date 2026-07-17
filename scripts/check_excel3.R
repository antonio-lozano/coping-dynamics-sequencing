suppressPackageStartupMessages(library(readxl))

f2 <- "C:/Users/jenif/Downloads/Statistical_report.xlsx"

# Read full sheet to find sniffing column positions
sr <- read_excel(f2, sheet = "Fig.4A_Clusters_frequency", col_names = FALSE)
cat("Dimensions:", nrow(sr), "x", ncol(sr), "\n")

# Print first 2 rows to find column headers
cat("\nRow 1:\n")
print(as.character(unlist(sr[1, ])))
cat("\nRow 2:\n")
print(as.character(unlist(sr[2, ])))

# Find columns containing "Sniff" or "sniff"
cols <- as.character(unlist(sr[1, ]))
sniff_cols <- which(grepl("niff|nif", cols, ignore.case = TRUE))
cat("\nSniff-related columns:", sniff_cols, "\n")

# Print rows 1-10 for sniff columns
if (length(sniff_cols) > 0) {
  start <- max(1, sniff_cols[1] - 1)
  end   <- min(ncol(sr), sniff_cols[1] + 10)
  cat("\nSniff section (cols", start, "to", end, "):\n")
  print(sr[1:40, start:end])
}
