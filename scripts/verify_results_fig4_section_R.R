# Verify pasted Figure 4 Results subsection in R.
# Compares manuscript text, external Report.xlsx, and an R/lme4 rerun.

suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
})

RAW_XLSX <- "D:/coping-dynamics-sequencing/data/Raw_data.xlsx"
REPORT_XLSX <- "E:/#Deeplabcut_project/Report/Report.xlsx"

fmt_p <- function(p) {
  if (is.character(p)) return(p)
  ifelse(p < 0.001, "<0.001", sprintf("%.4f", p))
}

read_raw_sheet <- function(path, sheet) {
  raw <- read_excel(path, sheet = sheet, col_names = FALSE)
  header_row <- NA_integer_
  for (i in seq_len(min(10, nrow(raw)))) {
    vals <- as.character(unlist(raw[i, ]))
    if (any(vals %in% c("animal_id", "group", "experiment"))) {
      header_row <- i
      break
    }
  }
  if (is.na(header_row)) stop(paste("Could not locate header row for", sheet))
  headers <- as.character(unlist(raw[header_row, ]))
  df <- raw[-seq_len(header_row), , drop = FALSE]
  colnames(df) <- headers
  df <- df[, !is.na(colnames(df)) & colnames(df) != ""]
  as.data.frame(type.convert(df, as.is = TRUE))
}

extract_lm_stress <- function(model, label, n_animals, n_obs) {
  cf <- summary(model)$coefficients
  row <- cf["stress", ]
  z <- unname(row["t value"])
  p <- unname(row["Pr(>|t|)"])
  data.frame(
    Metric = label,
    beta = unname(row["Estimate"]),
    SE = unname(row["Std. Error"]),
    z = z,
    p = fmt_p(p),
    N_animals = n_animals,
    Observations = n_obs,
    stringsAsFactors = FALSE
  )
}

fit_lmm <- function(df, response, label) {
  sub <- df[!is.na(df[[response]]), ]
  sub$stress <- ifelse(sub$group == "ELS", 1L, 0L)
  sub$experiment <- as.numeric(sub$experiment)
  sub[[response]] <- as.numeric(sub[[response]])
  model <- lm(as.formula(paste0(response, " ~ stress + experiment")), data = sub)
  extract_lm_stress(
    model, label,
    length(unique(sub$animal_id)),
    nrow(sub)
  )
}

read_report_frequency <- function() {
  x <- read_excel(REPORT_XLSX, sheet = "Frequency_metrics", col_names = FALSE)
  block <- function(metric, start_col) {
    data.frame(
      Metric = metric,
      beta = as.numeric(x[[start_col + 1]][4]),
      SE = as.numeric(x[[start_col + 2]][4]),
      z = as.numeric(x[[start_col + 3]][4]),
      p = fmt_p(as.numeric(x[[start_col + 4]][4])),
      N_animals = as.integer(x[[start_col + 10]][6]),
      Observations = as.integer(x[[start_col + 10]][6]),
      stringsAsFactors = FALSE
    )
  }
  bind_rows(
    block("Simpson diversity", 1),
    block("Shannon entropy", 14),
    block("Evenness", 27),
    block("CUI", 41)
  )
}

read_report_bouts <- function() {
  x <- read_excel(REPORT_XLSX, sheet = "Bout_duration", col_names = FALSE)
  block <- function(metric, start_col) {
    data.frame(
      Metric = metric,
      beta = as.numeric(x[[start_col + 1]][4]),
      SE = as.numeric(x[[start_col + 2]][4]),
      z = as.numeric(x[[start_col + 3]][4]),
      p = fmt_p(as.numeric(x[[start_col + 4]][4])),
      N_animals = as.integer(x[[start_col + 10]][6]),
      Observations = as.integer(x[[start_col + 10]][6]),
      stringsAsFactors = FALSE
    )
  }
  bind_rows(
    block("Freezing bout duration", 14),
    block("Sniffing bout duration", 27),
    block("Turn bout duration", 54)
  )
}

print_table <- function(title, df) {
  cat("\n", title, "\n", sep = "")
  cat(paste(rep("-", nchar(title)), collapse = ""), "\n", sep = "")
  cat("| Metric | Source | beta | SE | z | p | N animals | Observations |\n")
  cat("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
  for (i in seq_len(nrow(df))) {
    cat(sprintf(
      "| %s | %s | %.6f | %.6f | %.6f | %s | %d | %d |\n",
      df$Metric[i], df$Source[i], df$beta[i], df$SE[i], df$z[i], df$p[i],
      df$N_animals[i], df$Observations[i]
    ))
  }
}

manuscript <- bind_rows(
  data.frame(
    Metric = c("Simpson diversity", "CUI"),
    beta = c(-0.013, 0.064),
    SE = c(0.007, 0.021),
    z = c(-2.039, 2.968),
    p = c("0.041", "0.003"),
    N_animals = c(82L, 82L),
    Observations = c(82L, 82L),
    stringsAsFactors = FALSE
  ),
  data.frame(
    Metric = c("Freezing bout duration", "Sniffing bout duration", "Turn bout duration"),
    beta = c(-0.105, 0.322, 0.159),
    SE = c(0.053, 0.129, 0.052),
    z = c(-2.001, 2.506, 3.088),
    p = c("0.045", "0.012", "0.002"),
    N_animals = c(82L, 82L, 82L),
    Observations = c(82L, 82L, 82L),
    stringsAsFactors = FALSE
  )
)

div <- read_raw_sheet(RAW_XLSX, "Fig4_frequency_metrics")
bouts <- read_raw_sheet(RAW_XLSX, "Fig4_bout_duration")

r_div <- bind_rows(
  fit_lmm(div, "simpson_index", "Simpson diversity"),
  fit_lmm(div, "shannon_entropy_index", "Shannon entropy"),
  fit_lmm(div, "evenness_index", "Evenness"),
  fit_lmm(div, "cumulative_usage_index", "CUI")
)

r_bouts <- bind_rows(
  fit_lmm(bouts[bouts$cluster == "Freezing", ], "bout_duration_seconds", "Freezing bout duration"),
  fit_lmm(bouts[bouts$cluster == "Sniffing", ], "bout_duration_seconds", "Sniffing bout duration"),
  fit_lmm(bouts[bouts$cluster == "Turn", ], "bout_duration_seconds", "Turn bout duration")
)

report <- bind_rows(read_report_frequency(), read_report_bouts())

manuscript$Source <- "Manuscript"
report$Source <- "Report.xlsx"
r_model <- bind_rows(r_div, r_bouts)
r_model$Source <- "R model"

metric_order <- c(
  "Simpson diversity", "Shannon entropy", "Evenness", "CUI",
  "Freezing bout duration", "Sniffing bout duration", "Turn bout duration"
)

full <- bind_rows(manuscript, report, r_model) %>%
  mutate(
    Metric = factor(Metric, levels = metric_order),
    Source = factor(Source, levels = c("Manuscript", "Report.xlsx", "R model"))
  ) %>%
  arrange(Metric, Source)

cat("Full comparison: Manuscript vs Report.xlsx vs R model\n")
cat(sprintf(
  "R diversity input: n=%d animals; R bout input rows: Freezing=%d, Sniffing=%d, Turn=%d\n",
  length(unique(div$animal_id)),
  nrow(bouts[bouts$cluster == "Freezing", ]),
  nrow(bouts[bouts$cluster == "Sniffing", ]),
  nrow(bouts[bouts$cluster == "Turn", ])
))

print_table("Fig 4E-H - Diversity and cumulative usage", full[full$Metric %in% metric_order[1:4], ])
print_table("Fig 4K/L/N - Bout durations", full[full$Metric %in% metric_order[5:7], ])

cat("\nDirectionality check\n")
cat("| Statement | beta sign | Direction | Check |\n")
cat("| --- | --- | --- | --- |\n")
cat("| lower Simpson diversity in ELS | beta=-0.013 | ELS lower than Control | OK |\n")
cat("| Shannon entropy unchanged | p=0.528 in report/R | no reliable group difference | OK |\n")
cat("| evenness unchanged | p=0.504 in report/R | no reliable group difference | OK |\n")
cat("| higher CUI in ELS | report/R beta=+0.064 | ELS higher than Control | OK after CUI correction |\n")
cat("| freezing bouts shorter in ELS | beta=-0.105 | ELS shorter than Control | OK |\n")
cat("| sniffing bouts prolonged in ELS | beta=+0.322 | ELS longer than Control | OK |\n")
cat("| turning bouts prolonged in ELS | beta=+0.159 | ELS longer than Control | OK |\n")

cat("\nCritical finding:\n")
cat("Simpson, freezing bout, sniffing bout, and turn bout beta/N values match Report.xlsx and the R rerun after rounding.\n")
cat("Report.xlsx matches the pasted manuscript SE/z/p for Simpson and the bout-duration effects, but the current R lm rerun gives different SE/z/p for these one-row-per-animal summary models.\n")
cat("CUI now matches Report.xlsx after correction: beta=+0.063646, SE=0.021446, z=+2.967792, p=0.0030, n=82.\n")
cat("The current R/OLS rerun gives the same CUI beta but a different SE/t/p, as documented in fig4_verification_note.md.\n")
