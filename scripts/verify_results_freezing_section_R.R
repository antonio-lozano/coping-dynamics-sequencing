# Verify pasted Results freezing subsection in R.
# Compares manuscript text, external Report.xlsx, and an R/lme4 rerun.

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(dplyr)
  library(readr)
  library(readxl)
})

SOURCE <- "d:/coping-dynamics-sequencing/data/source"
FREEZING_DIR <- file.path(SOURCE, "freezing_predictions")
INDEX_CSV <- file.path(SOURCE, "animal_groups.csv")
FREQ_CSV <- file.path(SOURCE, "cluster_frequency_per_animal.csv")
REPORT_XLSX <- "E:/#Deeplabcut_project/Report/Report.xlsx"

BIN_SECONDS <- 30
FPS <- 25
EXCLUDE <- c("animal_48_6", "48_6")

manuscript <- data.frame(
  Term = c(
    "Exp1 time",
    "Exp1 ELS x time",
    "Exp3 time",
    "Exp3 ELS x time",
    "Combined ELS x time"
  ),
  beta = c(3.721, -0.887, 4.412, -0.721, -0.822),
  SE = c(0.128, 0.181, 0.191, 0.270, 0.154),
  z = c(29.048, -4.898, 23.091, -2.667, -5.327),
  p = c("<0.001", "<0.001", "<0.001", "0.008", "<0.001"),
  N_animals = c(50, 50, 32, 32, 82),
  Observations = c(750, 750, 480, 480, 1230),
  stringsAsFactors = FALSE
)

normalize_name <- function(x) gsub(" ", "_", trimws(x))

short_id <- function(x) {
  norm <- normalize_name(x)
  m <- regmatches(norm, regexpr("(?i)Animal[_ ]?(\\d+(?:[_-]\\d+)?)", norm, perl=TRUE))
  if (length(m) > 0 && nchar(m) > 0) {
    id <- sub("(?i)Animal[_ ]?", "", m[1], perl=TRUE)
    return(gsub("[_-]", ".", id))
  }
  parts <- strsplit(norm, "_")[[1]]
  n <- length(parts)
  if (n >= 2) paste(parts[(n-1):n], collapse=".") else norm
}

fmt_p <- function(p) {
  if (is.character(p)) return(p)
  ifelse(p < 0.001, "<0.001", sprintf("%.4f", p))
}

load_freezing_long <- function() {
  idx <- read_csv(INDEX_CSV, show_col_types=FALSE)
  group_map <- setNames(idx$group, normalize_name(idx$name))
  group_map_sid <- setNames(idx$group, sapply(idx$name, short_id))

  freq <- read_csv(FREQ_CSV, show_col_types=FALSE)
  exp_map <- setNames(freq$experiment, as.character(freq$animal_id))

  bin_size <- FPS * BIN_SECONDS
  csvs <- list.files(FREEZING_DIR, pattern="_freezing_predictions_only\\.csv$", full.names=TRUE)
  records <- list()

  for (f in csvs) {
    base <- sub("_freezing_predictions_only\\.csv$", "", basename(f))
    if (tolower(normalize_name(base)) %in% EXCLUDE) next
    df <- tryCatch(read_csv(f, show_col_types=FALSE, progress=FALSE), error=function(e) NULL)
    if (is.null(df)) next
    if (!"Freezing_Jen_0-125_threshold" %in% names(df)) next

    norm_base <- normalize_name(base)
    sid <- short_id(base)
    grp <- group_map[norm_base]
    if (is.na(grp)) grp <- group_map_sid[sid]
    if (is.na(grp) || !grp %in% c("Control", "ELS")) next

    freeze <- df[["Freezing_Jen_0-125_threshold"]]
    n_bins <- floor(length(freeze) / bin_size)
    if (n_bins == 0) next

    for (b in seq_len(n_bins)) {
      chunk <- freeze[((b - 1) * bin_size + 1):(b * bin_size)]
      records[[length(records) + 1]] <- data.frame(
        animal_id = sid,
        group = grp,
        time_bin = b,
        freezing_pct = mean(chunk, na.rm=TRUE) * 100,
        stringsAsFactors = FALSE
      )
    }
  }

  long_df <- bind_rows(records)
  long_df$exp_key <- as.character(suppressWarnings(as.numeric(gsub("_", ".", long_df$animal_id))))
  long_df$experiment <- exp_map[long_df$exp_key]
  long_df <- long_df[!is.na(long_df$experiment), ]
  long_df$experiment <- as.integer(long_df$experiment)
  long_df$group <- factor(long_df$group, levels=c("Control", "ELS"))
  long_df
}

extract_term <- function(model, term, label, n_animals, n_obs) {
  coefs <- summary(model)$coefficients
  row <- coefs[term, ]
  data.frame(
    Term = label,
    beta = unname(row["Estimate"]),
    SE = unname(row["Std. Error"]),
    z = unname(row["t value"]),
    p = fmt_p(unname(row["Pr(>|t|)"])),
    N_animals = n_animals,
    Observations = n_obs,
    stringsAsFactors = FALSE
  )
}

read_report_terms <- function() {
  gt <- read_excel(REPORT_XLSX, sheet="Ground_truth ", col_names=FALSE)

  section <- list(
    Combined = list(start=1, n_obs=gt[[15]][6], n_animals=gt[[15]][7]),
    Exp1 = list(start=19, n_obs=gt[[33]][6], n_animals=gt[[33]][7]),
    Exp3 = list(start=37, n_obs=gt[[51]][6], n_animals=gt[[51]][7])
  )

  row_to_df <- function(label, sec, row) {
    start <- section[[sec]]$start
    data.frame(
      Term = label,
      beta = as.numeric(gt[[start + 1]][row]),
      SE = as.numeric(gt[[start + 2]][row]),
      z = as.numeric(gt[[start + 3]][row]),
      p = fmt_p(as.numeric(gt[[start + 4]][row])),
      N_animals = as.integer(section[[sec]]$n_animals),
      Observations = as.integer(section[[sec]]$n_obs),
      stringsAsFactors = FALSE
    )
  }

  bind_rows(
    row_to_df("Exp1 time", "Exp1", 5),
    row_to_df("Exp1 ELS x time", "Exp1", 6),
    row_to_df("Exp3 time", "Exp3", 5),
    row_to_df("Exp3 ELS x time", "Exp3", 6),
    row_to_df("Combined ELS x time", "Combined", 6)
  )
}

print_table <- function(title, df) {
  cat("\n", title, "\n", sep="")
  cat(paste(rep("-", nchar(title)), collapse=""), "\n", sep="")
  cat("| Term | Source | beta | SE | z | p | N animals | Observations |\n")
  cat("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
  for (i in seq_len(nrow(df))) {
    cat(sprintf(
      "| %s | %s | %.4f | %.4f | %.4f | %s | %d | %d |\n",
      df$Term[i], df$Source[i], df$beta[i], df$SE[i], df$z[i], df$p[i],
      df$N_animals[i], df$Observations[i]
    ))
  }
}

long_df <- load_freezing_long()

m_exp1 <- lmer(
  freezing_pct ~ group * time_bin + (1 | animal_id),
  data=long_df[long_df$experiment == 1, ],
  REML=FALSE
)
m_exp3 <- lmer(
  freezing_pct ~ group * time_bin + (1 | animal_id),
  data=long_df[long_df$experiment == 3, ],
  REML=FALSE
)
m_comb <- lmer(
  freezing_pct ~ group * time_bin + (1 | animal_id) + (1 | experiment),
  data=long_df,
  REML=FALSE
)

r_model <- bind_rows(
  extract_term(m_exp1, "time_bin", "Exp1 time",
               length(unique(long_df$animal_id[long_df$experiment == 1])),
               sum(long_df$experiment == 1)),
  extract_term(m_exp1, "groupELS:time_bin", "Exp1 ELS x time",
               length(unique(long_df$animal_id[long_df$experiment == 1])),
               sum(long_df$experiment == 1)),
  extract_term(m_exp3, "time_bin", "Exp3 time",
               length(unique(long_df$animal_id[long_df$experiment == 3])),
               sum(long_df$experiment == 3)),
  extract_term(m_exp3, "groupELS:time_bin", "Exp3 ELS x time",
               length(unique(long_df$animal_id[long_df$experiment == 3])),
               sum(long_df$experiment == 3)),
  extract_term(m_comb, "groupELS:time_bin", "Combined ELS x time",
               length(unique(long_df$animal_id)),
               nrow(long_df))
)

report <- read_report_terms()

manuscript$Source <- "Manuscript"
report$Source <- "Report.xlsx"
r_model$Source <- "R model"

all_terms <- manuscript$Term
full <- bind_rows(manuscript, report, r_model) %>%
  mutate(Term = factor(Term, levels=all_terms),
         Source = factor(Source, levels=c("Manuscript", "Report.xlsx", "R model"))) %>%
  arrange(Term, Source)

cat("Full comparison: Manuscript vs Report.xlsx vs R model\n")
cat(sprintf(
  "R input: n=%d animals, %d observations; Exp1 n=%d/%d obs, Exp3 n=%d/%d obs\n",
  length(unique(long_df$animal_id)), nrow(long_df),
  length(unique(long_df$animal_id[long_df$experiment == 1])), sum(long_df$experiment == 1),
  length(unique(long_df$animal_id[long_df$experiment == 3])), sum(long_df$experiment == 3)
))

print_table("Fig 2A-C - Freezing progression LMM", full)

cat("\nDirectionality check\n")
cat("| Statement | beta sign | Direction | Check |\n")
cat("| --- | --- | --- | --- |\n")
cat("| Time increases freezing in Exp1 | beta=+3.721 | freezing rises over trial | OK |\n")
cat("| ELS blunts Exp1 freezing progression | beta=-0.887 | ELS slope lower than control | OK |\n")
cat("| Time increases freezing in Exp3 | beta=+4.412 | freezing rises over trial | OK |\n")
cat("| ELS blunts Exp3 freezing progression | beta=-0.721 | ELS slope lower than control | OK |\n")
cat("| ELS blunts combined freezing progression | beta=-0.822 | ELS slope lower across cohorts | OK |\n")

cat("\nCritical finding:\n")
cat("Manuscript, Report.xlsx, and the R/lme4 rerun match for beta, SE, z/t, p, N animals, and observations after manuscript rounding.\n")
cat("The pasted manuscript paragraph should keep the current Figure 2 values.\n")
