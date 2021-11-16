import pandas as pd
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-predictions_file", "--predictions-file", required=True)
    parser.add_argument("-ground_truth_file", "--ground-truth-file", required=True)

    return parser.parse_args()

DEFAULT_BETA = 1.0
def main():
    args = parse_args()

    ground_truth = pd.read_csv(args.ground_truth_file)
    ground_truth.index = ground_truth['file']
    ground_truth = ground_truth.drop(columns=['file'])

    predictions = pd.read_csv(args.predictions_file)
    predictions.index = predictions['file']
    predictions = predictions.drop(columns=['file'])

    authors = set(ground_truth['author']) 
    confusion_matrix = pd.DataFrame(np.zeros((len(authors), len(authors))), index=authors, columns=authors)
    for file in predictions.index: 
        actual, predicted = ground_truth.loc[file, 'author'], predictions.loc[file, 'predicted_author']
        confusion_matrix.at[actual, predicted] += 1

    total_hits = 0
    total_incorrect = 0
    for author in authors: 
        print(f"author: {author}")
        hits = confusion_matrix.at[author, author]
        total_hits += hits
        print(f"\t hits: {hits}")
        strikes = confusion_matrix[author].sum() - hits
        print(f"\t strikes: {strikes} ")
        misses = confusion_matrix.loc[author].sum() - hits
        total_incorrect += strikes+misses 
        print(f"\t misses: {misses} ")
        precision = hits / (hits+strikes)
        recall = hits / (hits + misses)
        f1measure = (1+DEFAULT_BETA**2) * precision * recall / (DEFAULT_BETA**2 * precision + recall)
        print(f"precision: {precision}")
        print(f"recall: {recall}")
        print(f"f1: {f1measure}")
    
    total_entries = confusion_matrix.stack().sum()
    print(f"total correct: {total_hits}")
    print(f"total incorrect: {total_entries - total_hits}")
    print(f"total: {total_entries}")
    print(f"total accuracy: {total_hits / total_entries}")
    
    with open('confusion_matrix.csv', 'w+') as f:
        confusion_matrix.insert(0, column='true/predicted', value=confusion_matrix.index)
        confusion_matrix.to_csv(f, index=False)

if __name__ == "__main__":
    main()