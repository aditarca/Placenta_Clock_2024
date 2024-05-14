#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(optparse))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(magrittr))
suppressPackageStartupMessages(library(yardstick))

option_list <- list( 
    make_option(c('-p', '--predictions_file'),  type = "character"),
    make_option(c('-g', '--goldstandard_file'),  type = "character"),
    make_option(c('-o', '--output'),  type = "character", default="results.json"),
    make_option(c('-r', '--real_scores'),  type = "character", default="true_results.json"),
    make_option(c('-t', '--task'),  type = "character", default="1")
)
args <- parse_args(OptionParser(option_list=option_list))

# Set colname depending on task number.
colname <- ifelse(args$task == "1", "was_preterm", "was_early_preterm")

# Read in files and ensure participant order matches between the files.
gold <- read_csv(args$goldstandard_file, col_types=cols("c", "f", "d"))
pred <- read_csv(args$predictions_file, show_col_types=FALSE)
pred <- pred[match(gold$participant, pred$participant), ]

# Set factor levels (otherwise, they will be mismatched).
gold[[colname]] <- factor(gold[[colname]], levels=c(1,0))
pred[[colname]] <- factor(pred[[colname]], levels=c(1,0))

# Calculate the true scores first.
true_scores <- list(
    "auc_roc" = roc_auc_vec(gold[[colname]], pred$probability),
    "auprc" = pr_auc_vec(gold[[colname]], pred$probability),
    "accuracy" = accuracy_vec(gold[[colname]], pred[[colname]]),
    "sensitivity" = sens_vec(gold[[colname]], pred[[colname]]),
    "specificity" = spec_vec(gold[[colname]], pred[[colname]]),
    "mcc" = mcc_vec(gold[[colname]], pred[[colname]])
)
export_json <- toJSON(true_scores, auto_unbox = TRUE, pretty=T)
write(export_json, args$real_scores)

# Calculate bootstrapped scores, using a fixed seed value.
set.seed(7192022)
BS_n <- 100
bs_indices <- matrix(1:nrow(gold), nrow(gold), BS_n) %>%
    apply(2, sample, replace=T)

boot_auc_roc <- apply(bs_indices[1:37,], 2, function(ind) {
    tmp <- pred[match(gold$participant, pred$participant), ]
    suppressWarnings(roc_auc_vec(gold[[colname]][ind], tmp$probability[ind]))
})
boot_aupr <- apply(bs_indices[1:37,], 2, function(ind) {
    tmp <- pred[match(gold$participant, pred$participant), ]
    suppressWarnings(pr_auc_vec(gold[[colname]][ind], tmp$probability[ind]))
})
boot_acc.median <- apply(bs_indices[1:37,], 2, function(ind) {
    tmp <- pred[match(gold$participant, pred$participant), ]
    suppressWarnings(accuracy_vec(gold[[colname]][ind], tmp[[colname]][ind]))
}) %>% median(na.rm = TRUE)
boot_sens.median <- apply(bs_indices[1:37,], 2, function(ind) {
    tmp <- pred[match(gold$participant, pred$participant), ]
    suppressWarnings(sens_vec(gold[[colname]][ind], tmp[[colname]][ind]))
}) %>% median(na.rm = TRUE)
boot_spec.median <- apply(bs_indices[1:37,], 2, function(ind) {
    tmp <- pred[match(gold$participant, pred$participant), ]
    suppressWarnings(spec_vec(gold[[colname]][ind], tmp[[colname]][ind]))
}) %>% median(na.rm = TRUE)

bs_scores <- list(
    "boot_auc_roc" = boot_auc_roc %>% median(na.rm = TRUE),
    "boot_auc_roc_sd" = boot_auc_roc %>% sd(na.rm = TRUE),
    "boot_auc_roc_iqr" = boot_auc_roc %>% IQR(na.rm = TRUE),
    "boot_auprc" = boot_aupr %>% median(na.rm = TRUE),
    "boot_auprc_sd" = boot_aupr %>% sd(na.rm = TRUE),
    "boot_auprc_iqr" = boot_auc_roc %>% IQR(na.rm = TRUE),
    "boot_accuracy" = boot_acc.median,
    "boot_sensitivity" = boot_sens.median,
    "boot_specificity" = boot_spec.median,
    "submission_status" = "SCORED"
)
export_json <- toJSON(bs_scores, auto_unbox = TRUE, pretty=T)
write(export_json, args$output)

