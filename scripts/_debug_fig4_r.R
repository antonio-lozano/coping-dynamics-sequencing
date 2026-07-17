# Try different R models to reproduce Report.xlsx SE for simpson
library(lme4)

repo <- "d:/coping-dynamics-sequencing"
sys_path <- file.path(repo, "data/source/syllable_usage_per_timebin_250ms.csv")
raw <- read.csv(sys_path)
colnames(raw)[colnames(raw) == "Time.Bin"] <- "time_bin"
colnames(raw)[colnames(raw) == "Condition"] <- "group"
raw <- raw[raw$group %in% c("Control", "ELS"), ]
raw$Animal <- as.character(raw$Animal)
raw$Syllable <- as.integer(raw$Syllable)

# Cluster map
cluster_map <- list(
  Freezing=c(0,28), Sniffing=c(18,20), Grooming=c(24),
  Turn=c(1,3,5,6,10,15,26,27), Locomotion=c(11,12,14,16,19,21,25),
  Climbing=c(111), Jump=c(23,29,30,34)
)
s2c <- setNames(rep(names(cluster_map), sapply(cluster_map, length)), unlist(cluster_map))
raw$cluster <- s2c[as.character(raw$Syllable)]
raw$cluster[is.na(raw$cluster)] <- ""

# Dominant cluster per time bin
pred <- do.call(rbind, lapply(split(raw, list(raw$Animal, raw$time_bin)), function(g) {
  g[which.max(g$Percentage), ]
}))
meta <- unique(pred[, c("Animal", "group", "Experiment")])

# Full sequences
full_seq <- tapply(raw$cluster, raw$Animal, function(x) x)

# Compute simpson per animal
compute_div <- function(seqs, meta_df) {
  rows <- lapply(names(seqs), function(a) {
    seq <- seqs[[a]]
    info <- meta_df[meta_df$Animal == a, ]
    tab <- table(seq)
    p <- as.numeric(tab) / sum(tab)
    simpson <- 1 - sum(p^2)
    p_nz <- p[p > 0]
    shannon <- -sum(p_nz * log(p_nz))
    evenness <- if (length(p_nz) > 1) shannon / log(length(p_nz)) else NA
    # CUI
    named_p <- p[names(p) != ""]
    named_p <- named_p / sum(named_p)
    sorted_p <- sort(named_p, decreasing=TRUE)
    cum <- cumsum(sorted_p)
    baseline <- (length(cum) + 1) / (2 * length(cum))
    cui <- (mean(cum) - baseline) / (1 - baseline)
    data.frame(Animal=a, group=info$group, Experiment=as.character(info$Experiment),
               simpson=simpson, shannon=shannon, evenness=evenness, cui=cui)
  })
  do.call(rbind, rows)
}
df <- compute_div(full_seq, meta)
df$Stress <- ifelse(df$group == "ELS", 1, 0)
df$Experiment <- factor(df$Experiment)
cat("n animals:", nrow(df), "\n")
cat("groups:", table(df$group), "\n")

cat("\n=== MODEL COMPARISONS FOR SIMPSON ===\n")
cat("Gold: beta=-0.01344087 SE=0.00659293 z=-2.03867875\n\n")

# A: lm(simpson ~ group + Experiment)
m_a <- lm(simpson ~ group + Experiment, data=df)
coef_a <- coef(summary(m_a))
cat("A: lm(~group+Experiment): beta=", coef_a["groupELS","Estimate"],
    "SE=", coef_a["groupELS","Std. Error"],
    "t=", coef_a["groupELS","t value"], "\n")

# B: lmer(simpson ~ group + (1|Experiment))
m_b <- lmer(simpson ~ group + (1|Experiment), data=df, REML=FALSE)
coef_b <- coef(summary(m_b))
cat("B: lmer(~group+(1|Exp)): beta=", coef_b["groupELS","Estimate"],
    "SE=", coef_b["groupELS","Std. Error"],
    "t=", coef_b["groupELS","t value"], "\n")

# C: lmer(simpson ~ group + (1|Experiment), REML=TRUE)
m_c <- lmer(simpson ~ group + (1|Experiment), data=df, REML=TRUE)
coef_c <- coef(summary(m_c))
cat("C: lmer(~group+(1|Exp), REML): beta=", coef_c["groupELS","Estimate"],
    "SE=", coef_c["groupELS","Std. Error"],
    "t=", coef_c["groupELS","t value"], "\n")

# D: lmer(simpson ~ group + (1|Animal) + (0+1|Experiment)) -- same as combined timecourse
# But with 1 obs/animal the animal random effect is singular
m_d <- tryCatch(lmer(simpson ~ group + (1|Animal) + (0+1|Experiment), data=df, REML=FALSE), error=function(e) NULL)
if (!is.null(m_d)) {
  coef_d <- coef(summary(m_d))
  cat("D: lmer(~group+(1|animal)+(0+1|Exp)): beta=", coef_d["groupELS","Estimate"],
      "SE=", coef_d["groupELS","Std. Error"],
      "t=", coef_d["groupELS","t value"], "\n")
}

# E: lm(simpson ~ group) only
m_e <- lm(simpson ~ group, data=df)
coef_e <- coef(summary(m_e))
cat("E: lm(~group only): beta=", coef_e["groupELS","Estimate"],
    "SE=", coef_e["groupELS","Std. Error"],
    "t=", coef_e["groupELS","t value"], "\n")

cat("\n=== CUI ===\n")
cat("Gold: beta=0.06364579 SE=0.02144550 z=2.96779168\n")
m_cui_a <- lm(cui ~ group + Experiment, data=df)
coef_cui <- coef(summary(m_cui_a))
cat("A: lm(cui~group+Exp): beta=", coef_cui["groupELS","Estimate"],
    "SE=", coef_cui["groupELS","Std. Error"],
    "t=", coef_cui["groupELS","t value"], "\n")
m_cui_b <- lmer(cui ~ group + (1|Experiment), data=df, REML=FALSE)
coef_cui_b <- coef(summary(m_cui_b))
cat("B: lmer(cui~group+(1|Exp)): beta=", coef_cui_b["groupELS","Estimate"],
    "SE=", coef_cui_b["groupELS","Std. Error"],
    "t=", coef_cui_b["groupELS","t value"], "\n")
