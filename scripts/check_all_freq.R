suppressPackageStartupMessages(library(glmmTMB))

freq_df <- read.csv("D:/coping-dynamics-sequencing/data/source/cluster_frequency_per_animal.csv")
freq_df$stress     <- ifelse(freq_df$group == "ELS", 1L, 0L)
freq_df$experiment <- as.factor(as.character(freq_df$experiment))

clusters <- c("Freeze", "Sniff", "Turn", "Locomotion")

cat(sprintf("%-12s  %8s  %8s  %8s  %8s\n", "Cluster", "beta", "SE", "z", "p"))
cat(strrep("-", 56), "\n")

for (cl in clusters) {
  sub <- freq_df[freq_df$cluster == cl, ]
  sub$freq_frames <- round(sub$frequency_seconds * 25)
  m <- glmmTMB(freq_frames ~ stress + (1|experiment), data = sub, family = nbinom2)
  cf <- summary(m)$coefficients$cond
  cat(sprintf("%-12s  %8.4f  %8.4f  %8.4f  %8.4f\n",
      cl,
      cf["stress","Estimate"],
      cf["stress","Std. Error"],
      cf["stress","z value"],
      cf["stress","Pr(>|z|)"]))
}
