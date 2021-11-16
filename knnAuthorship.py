import pandas as pd
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ground_truth_file", "--ground-truth-file", required=True)
    parser.add_argument("-okapi", "--okapi", type=str)
    parser.add_argument("-cosine", "--cosine", type=str)
    parser.add_argument("-raw", "--raw", type=str)
    parser.add_argument("-k", type=int, required=True)

    return parser.parse_args()


K2 = 100
def generate_similarity_matrix_okapi(okapi_path, raw_freq_path):
    df_okapi = pd.read_csv(okapi_path)
    df_okapi.set_index('file', inplace=True)

    df_raw_freq = pd.read_csv(raw_freq_path)
    df_raw_freq.set_index('file', inplace=True)

    raw_arr = np.array(df_raw_freq)
    okapi_arr = np.array(df_okapi)
    normalized_raw = (K2+1)*raw_arr / (K2 + raw_arr)
    rows = np.matmul(okapi_arr, normalized_raw.T)

    return pd.DataFrame(rows, index=df_okapi.index, columns=df_okapi.index)

def generate_similarity_matrix_cosine(tfidf_path):
    tfidf = pd.read_csv(tfidf_path)
    tfidf.set_index('file', inplace=True)

    dotted = np.dot(tfidf, np.transpose(tfidf))
    row_norms = np.linalg.norm(tfidf, axis=1)
    norms = np.array([np.full(len(tfidf.index), np.linalg.norm(tfidf.loc[row])) for row in tfidf.index]) * row_norms 
    sim_matrix = pd.DataFrame(dotted/norms, index=tfidf.index,columns=tfidf.index)
    return sim_matrix

def knearest(sim_matrix, file_name, ground_truth, k):
    neighbor_idxs = np.argsort(sim_matrix.loc[file_name].values)[-(k+1):]
    neighbors = sim_matrix.columns[neighbor_idxs]
    neighbors = neighbors[neighbors != file_name]
    plurality = ground_truth.loc[neighbors].value_counts().idxmax()[0]
    return plurality

def main():
    args = parse_args()

    ground_truth = pd.read_csv(args.ground_truth_file)
    ground_truth.set_index('file', inplace=True)

    if args.okapi is not None:
        if args.raw is None:
            raise ValueError('raw frequency vector file must be supplied')
        sim_matrix = generate_similarity_matrix_okapi(args.okapi, args.raw)
    elif args.cosine is not None:
        sim_matrix = generate_similarity_matrix_cosine(args.cosine)

    predictions = pd.DataFrame(index=ground_truth.index, columns=['predicted_author'])
    for file_name in sim_matrix.index:
        predicted_author = knearest(sim_matrix, file_name, ground_truth, args.k)
        predictions.at[file_name, 'predicted_author'] = predicted_author
    
    with open('predictions.csv', 'w+') as f:
        predictions.insert(0, column='file', value=predictions.index)
        predictions.to_csv(f, index=False)
    
if __name__ == "__main__":
    main()