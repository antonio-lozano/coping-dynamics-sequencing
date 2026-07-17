# Fig 2A-C freezing LMM verification using lme4/lmerTest
# Matches original manuscript analysis (R-based)
#
# Manuscript claims:
#   Fig 2A (Exp1): time beta=3.768, SE=0.142, z=26.489, p<0.001
#                  ELS x time beta=-0.887, SE=0.181, z=-4.898, p<0.001
#   Fig 2B (Exp3): time beta=4.412, SE=0.191, z=23.091, p<0.001
#                  ELS x time beta=-0.721, SE=0.270, z=-2.667, p=0.008
#   Combined:      ELS x time beta=-0.822, SE=0.154, z=-5.327, p<0.001
#
# Report.xlsx Ground_truth sheet:
#   Exp1 time: beta=3.721, SE=0.128, z=29.048  <-- differs from manuscript
#   Exp3 time: beta=4.412, SE=0.191, z=23.091  matches
#   Combined ELS x time: beta=-0.822, SE=0.154, z=-5.327  matches

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(dplyr)
  library(readr)
})

# Resolve repo root from this script's own location (portable across machines)
.args <- commandArgs(trailingOnly = FALSE)
.script_path <- sub("^--file=", "", .args[grep("^--file=", .args)])
REPO_ROOT <- normalizePath(file.path(dirname(.script_path), ".."))
SOURCE <- file.path(REPO_ROOT, "data", "source")
FREEZING_DIR <- file.path(SOURCE, "freezing_predictions")
INDEX_CSV <- file.path(SOURCE, "animal_groups.csv")
FREQ_CSV  <- file.path(SOURCE, "cluster_frequency_per_animal.csv")
BIN_SECONDS <- 30
FPS <- 25  # actual recording FPS for freezing data (11250 frames / 7.5 min / 60s = 25 fps)

# Helpers
normalize_name <- function(x) gsub(" ", "_", trimws(x))

short_id <- function(x) {
  norm <- normalize_name(x)
  m <- regmatches(norm, regexpr("(?i)Animal[_ ]?(\\d+(?:[_-]\\d+)?)", norm, perl=TRUE))
  if (length(m) > 0 && nchar(m) > 0) {
    id <- sub("(?i)Animal[_ ]?", "", m, perl=TRUE)
    return(gsub("[_-]", ".", id))
  }
  parts <- strsplit(norm, "_")[[1]]
  n <- length(parts)
  if (n >= 2) paste(parts[(n-1):n], collapse=".") else norm
}

# Group map
idx <- read_csv(INDEX_CSV, show_col_types=FALSE)
group_map <- setNames(idx$group, normalize_name(idx$name))
group_map_sid <- setNames(idx$group, sapply(idx$name, short_id))

# Experiment map
freq <- read_csv(FREQ_CSV, show_col_types=FALSE)
exp_map <- setNames(freq$experiment, as.character(freq$animal_id))

# Load and bin freezing CSVs
bin_size <- FPS * BIN_SECONDS
csvs <- list.files(FREEZING_DIR, pattern="_freezing_predictions_only\\.csv$", full.names=TRUE)
records <- list()
for (f in csvs) {
  base <- sub("_freezing_predictions_only\\.csv$", "", basename(f))
  if (tolower(normalize_name(base)) %in% c("animal_48_6", "48_6")) next
  df <- tryCatch(read_csv(f, show_col_types=FALSE), error=function(e) NULL)
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
    chunk <- freeze[((b-1)*bin_size + 1):(b*bin_size)]
    records[[length(records)+1]] <- data.frame(
      animal_id = sid,
      group = grp,
      time_bin = b,
      freezing_pct = mean(chunk, na.rm=TRUE) * 100,
      stringsAsFactors = FALSE
    )
  }
}
long_df <- bind_rows(records)

# Experiment assignment
long_df$exp_key <- as.character(
  suppressWarnings(as.numeric(gsub("_", ".", long_df$animal_id)))
)
long_df$experiment <- exp_map[long_df$exp_key]
missing <- sum(is.na(long_df$experiment))
if (missing > 0) cat(sprintf("WARNING: %d rows with unknown experiment\n", missing))
long_df <- long_df[!is.na(long_df$experiment), ]
long_df$experiment <- as.integer(long_df$experiment)

cat(sprintf("Data: %d animals, %d observations\n",
            length(unique(long_df$animal_id)), nrow(long_df)))
cat(sprintf("Exp1: %d animals, %d obs\n",
            length(unique(long_df$animal_id[long_df$experiment==1])),
            sum(long_df$experiment==1)))
cat(sprintf("Exp3: %d animals, %d obs\n\n",
            length(unique(long_df$animal_id[long_df$experiment==3])),
            sum(long_df$experiment==3)))

long_df$group <- factor(long_df$group, levels=c("Control","ELS"))

fit_and_report <- function(sub, label) {
  cat(sprintf("=== %s (n=%d animals, %d obs) ===\n",
              label, length(unique(sub$animal_id)), nrow(sub)))
  m <- lmer(freezing_pct ~ group * time_bin + (1|animal_id), data=sub, REML=FALSE)
  coefs <- summary(m)$coefficients
  params <- c("time_bin", "groupELS:time_bin")
  for (p in params) {
    if (p %in% rownames(coefs)) {
      r <- coefs[p,]
      cat(sprintf("  %-35s beta=%7.4f  SE=%6.4f  z=%8.4f  p=%s\n",
                  p, r["Estimate"], r["Std. Error"], r["t value"],
                  ifelse(r["Pr(>|t|)"] < 0.001, "<0.001", sprintf("%.4f", r["Pr(>|t|)"]))))
    }
  }
  cat("\n")
}

fit_and_report(long_df[long_df$experiment==1,], "Exp1 (Sanguino-Gomez & Krugers 2024)")
fit_and_report(long_df[long_df$experiment==3,], "Exp3 (Sanguino-Gomez et al 2024)")

# Combined: add experiment as random effect
cat("=== Combined (n=", length(unique(long_df$animal_id)), "animals,", nrow(long_df), "obs) ===\n")
m_comb <- lmer(freezing_pct ~ group * time_bin + (1|animal_id) + (1|experiment),
               data=long_df, REML=FALSE)
coefs_c <- summary(m_comb)$coefficients
for (p in c("time_bin", "groupELS:time_bin")) {
  if (p %in% rownames(coefs_c)) {
    r <- coefs_c[p,]
    cat(sprintf("  %-35s beta=%7.4f  SE=%6.4f  z=%8.4f  p=%s\n",
                p, r["Estimate"], r["Std. Error"], r["t value"],
                ifelse(r["Pr(>|t|)"] < 0.001, "<0.001", sprintf("%.4f", r["Pr(>|t|)"]))))
  }
}

cat("\n--- MANUSCRIPT vs REPORT COMPARISON ---\n")
cat("Fig 2A Exp1 time:\n")
cat("  Manuscript: beta=3.768, SE=0.142, z=26.489\n")
cat("  Report.xlsx: beta=3.721, SE=0.128, z=29.048  <-- DISCREPANCY\n")
cat("Fig 2A Exp1 ELS x time:\n")
cat("  Manuscript: beta=-0.887, SE=0.181, z=-4.898\n")
cat("  Report.xlsx: beta=-0.887, SE=0.181, z=-4.898  [MATCH]\n")
cat("Fig 2B Exp3 time:\n")
cat("  Manuscript: beta=4.412, SE=0.191, z=23.091\n")
cat("  Report.xlsx: beta=4.412, SE=0.191, z=23.091  [MATCH]\n")
cat("Fig 2B Exp3 ELS x time:\n")
cat("  Manuscript: beta=-0.721, SE=0.270, z=-2.667, p=0.008\n")
cat("  Report.xlsx: beta=-0.721, SE=0.270, z=-2.667, p=0.008  [MATCH]\n")
cat("Combined ELS x time:\n")
cat("  Manuscript: beta=-0.822, SE=0.154, z=-5.327\n")
cat("  Report.xlsx: beta=-0.822, SE=0.154, z=-5.327  [MATCH]\n")
