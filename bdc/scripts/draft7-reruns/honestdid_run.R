# #172 Rambachan-Roth (2023) relative-magnitude bounds on the registered chain-linked indices (Appendix B).
# Run on a PC or Posit Cloud from this folder:  Rscript honestdid_run.R   (writes honestdid_results.txt)
# Inputs from r7_07_honestdid_input.py: honestdid_<Index>_<outcome>_betahat.csv and _sigma.csv (base Sep 2025 dropped).
# Alpha 0.10 (90% robust intervals, as the percentile-bootstrap breakdown in the draft). Targets: Mar 2026 and Jun 2026.
if (!requireNamespace("HonestDiD", quietly = TRUE)) {
  if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes", repos = "https://cloud.r-project.org")
  remotes::install_github("asheshrambachan/HonestDiD")
}
library(HonestDiD)
Mbar <- seq(0, 3, by = 0.05)
out <- file("honestdid_results.txt", "w")
w <- function(...) { s <- paste0(...); cat(s, "\n"); writeLines(s, out) }
w("HonestDiD relative magnitudes, alpha 0.10, run ", format(Sys.time()), ", HonestDiD ", as.character(packageVersion("HonestDiD")))
for (ix in c("Model", "Fund")) for (y in c("mark", "below99")) {
  b <- read.csv(sprintf("honestdid_%s_%s_betahat.csv", ix, y))
  S <- as.matrix(read.csv(sprintf("honestdid_%s_%s_sigma.csv", ix, y), header = FALSE))
  nPre <- sum(b$event_time < 0); nPost <- sum(b$event_time > 0)
  for (tg in c("2026-03", "2026-06")) {
    l <- as.numeric(b$quarter[b$event_time > 0] == tg)
    orig <- constructOriginalCS(betahat = b$betahat, sigma = S, numPrePeriods = nPre, numPostPeriods = nPost, l_vec = l, alpha = 0.10)
    rm <- createSensitivityResults_relativeMagnitudes(betahat = b$betahat, sigma = S, numPrePeriods = nPre,
            numPostPeriods = nPost, l_vec = l, Mbarvec = Mbar, alpha = 0.10)
    excl <- (rm$lb > 0) | (rm$ub < 0)
    bd <- if (!excl[1]) 0 else { k <- which(!excl)[1]; if (is.na(k)) max(rm$Mbar) else rm$Mbar[k - 1] }
    w(sprintf("%s %s target %s: estimate %+.2f, original 90%% CI [%+.2f, %+.2f]; breakdown Mbar %.2f", ix, y, tg,
              sum(l * b$betahat[b$event_time > 0]), orig$lb, orig$ub, bd))
    for (m in c(0.5, 1, 1.5, 2)) { r <- rm[abs(rm$Mbar - m) < 1e-9, ]; w(sprintf("   Mbar %.1f: [%+.2f, %+.2f]", m, r$lb, r$ub)) }
  }
}
close(out)
