suppressPackageStartupMessages(library(optparse))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(Metrics))
suppressPackageStartupMessages(library(tidyverse))


option_list <- list( 
  make_option(c('-p', '--predictions_file'), 
              type = "character", 
              help = "Path to the predictions file"),
  make_option(c('-g', '--goldstandard_file'), 
              type = "character", 
              help = "Path to the gold standard file"),
  make_option(c('-o', '--output'), 
              type = "character", 
              default = "results.json", 
              help = "Path to the output JSON file"),
  make_option(c('-r', '--real_scores'), 
              type = "character", 
              default = "true_results.json", 
              help = "Path to the true scores JSON file"),
  make_option(c('-t', '--task'), 
              type = "character", 
              default = "1", 
              help = "Task number")
)


args <- parse_args(OptionParser(option_list=option_list))

# Set colname 



#args$goldstandard_file<- "gold_s.csv"
#args$predictions_file<-  "pred.csv"
  
# Read in files and ensure participant order matches between the files.
gold <- read_csv(args$goldstandard_file, col_types=cols( "c", "d"))
pred <- read_csv(args$predictions_file, col_types=cols( "c", "d"))

# Ensure the order of participants matches between the files
pred <- pred[match(gold$ID, pred$ID), ]



# Calculate the true scores first.
true_scores <- list(
    "rmse" = Metrics::rmse(gold[["GA"]], pred[["GA_prediction"]]),
    "mae" =  Metrics::mae(gold[["GA"]], pred[["GA_prediction"]]),
    "cor" =  cor(gold[["GA"]], pred[["GA_prediction"]])
)

export_json <- toJSON(true_scores, auto_unbox = TRUE, pretty=T)
write(export_json, args$real_scores)


# Create a results object with submission status
results <- list(
  "submission_status" = "SCORED",
  "rmse" = true_scores$rmse,
  "mae" = true_scores$mae,
  "cor" = true_scores$cor
)

# Export the results to a JSON file
results_json <- toJSON(results, auto_unbox = TRUE, pretty = TRUE)
write(results_json, args$output)

