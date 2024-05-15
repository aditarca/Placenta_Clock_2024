#!/usr/bin/env python3
"""Validate prediction file.

Prediction files between Task 1 and 2 are the same, only the features allowed as predictors in teh model differ
"""

import argparse
import json

import pandas as pd
import numpy as np

COLS = {
    "1": {
        'Sample': str,
        'GA_prediction': np.float64
    },
    "2": {
        'Sample': str,
        'GA_prediction': np.float64
    }
}


def get_args():
    """Set up command-line interface and get arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--predictions_file",
                        type=str, required=True)
    parser.add_argument("-g", "--goldstandard_file",
                        type=str, required=True)
    parser.add_argument("-t", "--task", type=str, default="1")
    parser.add_argument("-o", "--output", type=str)
    return parser.parse_args()


def check_dups(pred):
    """Check for duplicate samples."""
    duplicates = pred.duplicated(subset=['Sample'])
    if duplicates.any():
        return (
            f"Found {duplicates.sum()} duplicate Sample(s): "
            f"{pred[duplicates].Sample.to_list()}"
        )
    return ""


def check_missing_ids(gold, pred):
    """Check for missing Samples."""
    pred = pred.set_index('Sample')
    missing_ids = gold.index.difference(pred.index)
    if missing_ids.any():
        return (
            f"Found {missing_ids.shape[0]} missing Sample ID(s): "
            f"{missing_ids.to_list()}"
        )
    return ""


def check_unknown_ids(gold, pred):
    """Check for unknown Sample IDs."""
    pred = pred.set_index('Sample')
    unknown_ids = pred.index.difference(gold.index)
    if unknown_ids.any():
        return (
            f"Found {unknown_ids.shape[0]} unknown Sample ID(s): "
            f"{unknown_ids.to_list()}"
        )
    return ""


def check_nan_values(pred):
    """Check for NAN predictions."""
    missing_probs = pred.GA_prediction.isna().sum()
    if missing_probs:
        return (
            f"'GA_prediction' column contains {missing_probs} NaN value(s)."
        )
    return ""


def check_prob_values(pred):
    """Check that GA are between [10, 43]."""
    if (pred.GA_prediction < 10).any() or (pred.GA_prediction > 43).any():
        return "'GA_prediction' column should be between [10, 43] inclusive."
    return ""


def validate(gold_file, pred_file, task_number):
    """Validate predictions file against goldstandard."""
    errors = []

    gold = pd.read_csv(gold_file,
                       index_col="Sample")
    try:
        pred = pd.read_csv(pred_file,
                           usecols=COLS[task_number],
                           dtype=COLS[task_number],
                           float_precision='round_trip')
    except ValueError as err:
        errors.append(
            f"Invalid column names and/or types: {str(err)}. "
            f"Expecting: {str(COLS[task_number])}."
        )
    else:
        errors.append(check_dups(pred))
        errors.append(check_missing_ids(gold, pred))
        errors.append(check_unknown_ids(gold, pred))
        errors.append(check_nan_values(pred))
        errors.append(check_binary_values(pred))
        errors.append(check_prob_values(pred))
    return errors


def main():
    """Main function."""
    args = get_args()

    invalid_reasons = validate(
        gold_file=args.goldstandard_file,
        pred_file=args.predictions_file,
        task_number=args.task
    )

    invalid_reasons = "\n".join(filter(None, invalid_reasons))
    status = "INVALID" if invalid_reasons else "VALIDATED"

    # truncate validation errors if >500 (character limit for sending email)
    if len(invalid_reasons) > 500:
        invalid_reasons = invalid_reasons[:496] + "..."
    res = json.dumps({
        "submission_status": status,
        "submission_errors": invalid_reasons
    })

    if args.output:
        with open(args.output, "w") as out:
            out.write(res)
    else:
        print(res)


if __name__ == "__main__":
    main()
