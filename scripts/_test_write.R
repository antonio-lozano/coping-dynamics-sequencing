suppressPackageStartupMessages(library(openxlsx))

# Create fresh workbook and test multi-row writing
wb <- createWorkbook()
addWorksheet(wb, "Test")
writeData(wb, "Test", "Title row", startRow=1, startCol=1, colNames=FALSE)
mergeCells(wb, "Test", rows=1, cols=1:5)
writeData(wb, "Test", "Row 2 col 1", startRow=2, startCol=1, colNames=FALSE)
writeData(wb, "Test", "Row 2 col 2", startRow=2, startCol=2, colNames=FALSE)
writeData(wb, "Test", "Row 3", startRow=3, startCol=1, colNames=FALSE)
saveWorkbook(wb, "d:/coping-dynamics-sequencing/results/statistical_reports/_test.xlsx", overwrite=TRUE)
cat("Saved test\n")

# Now check from R side
wb2 <- loadWorkbook("d:/coping-dynamics-sequencing/results/statistical_reports/_test.xlsx")
ws <- wb2[["Test"]]
cat("Sheets:", names(wb2), "\n")
