import pandas as pd
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_file", "--data-file", required=True)
    parser.add_argument("-ground_truth_file", "--ground-truth-file", required=True)
    parser.add_argument("-okapi", "--okapi", action="store_true")
    parser.add_argument("-cosine", "--cosine", action="store_true")
    parser.add_argument("-k", type=int, required=True)

    return parser.parse_args()

"""
sim_matrix:
     file1 file2 file3
file1 1    0.3 
file2        1
file3              1
"""
"""
file  author
file1 author
file2 author
"""
def knearest(sim_matrix, file_name, ground_truth, k):
    neighbor_idxs = np.argsort(sim_matrix.loc[file_name].values)[-(k+1):]
    neighbors = sim_matrix.columns[neighbor_idxs]
    neighbors = neighbors[neighbors != file_name]
    plurality = ground_truth.loc[neighbors].value_counts().idxmax()[0]
    return plurality

def main():
    args = parse_args()
    sim_matrix = pd.read_csv(args.data_file)
    sim_matrix.index = sim_matrix['name']
    sim_matrix = sim_matrix.drop(columns=['name'])

    ground_truth = pd.read_csv(args.ground_truth_file)
    ground_truth.index = ground_truth['file']
    ground_truth = ground_truth.drop(columns=['file'])

    predictions = pd.DataFrame(index=ground_truth.index, columns=['predicted_author'])
    for file_name in sim_matrix.index:
        predicted_author = knearest(sim_matrix, file_name, ground_truth, args.k)
        predictions.at[file_name, 'predicted_author'] = predicted_author
    
    with open('predictions.csv', 'w+') as f:
        predictions.insert(0, column='file', value=predictions.index)
        predictions.to_csv(f, index=False)
    
if __name__ == "__main__":
    main()