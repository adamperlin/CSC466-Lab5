import pandas as pd
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-okapi", "--okapi", type=str)
    parser.add_argument("-tfidf", "--tfidf", type=str)
    parser.add_argument("-raw", "--raw", type=str)

    return parser.parse_args()


K2 = 100


def generate_similarity_matrix_okapi(okapi_path, raw_freq_path):
    df_okapi = pd.read_csv(okapi_path)
    df_okapi.index = df_okapi["file"].values
    df_okapi = df_okapi.drop(columns=["file"])

    df_raw_freq = pd.read_csv(raw_freq_path)
    df_raw_freq.index = df_raw_freq["file"].values
    df_raw_freq = df_raw_freq.drop(columns=["file"])

    rows = []
    processed = 0
    for name in df_raw_freq.index:
        v = df_raw_freq.loc[name] * (K2 + 1)
        v = np.divide(v * (K2 + 1), v + K2)

        sim = np.sum(np.multiply(np.array(v), df_okapi), axis=1)

        rows.append(sim)
        print(f"\rprocessed: {processed} of {len(df_raw_freq.index)}")
        processed += 1

    return pd.DataFrame(rows, index=df_okapi.index, columns=df_okapi.index)


def generate_similarity_matrix_tfidf(tfidf_path):
    df = pd.read_csv(tfidf_path)
    df.index = df["file"].values
    df = df.drop(columns=["file"])

    rows = []
    processed = 0
    for name in df.index:
        v = df.loc[name]
        dots = np.dot(v, df.T)
        norm = np.linalg.norm(v) * np.linalg.norm(df, axis=1)
        rows.append(np.divide(dots, norm))
        print(f"\rprocessed: {processed} of {len(df.index)}")
        processed += 1

    return pd.DataFrame(rows, index=df.index, columns=df.index)


def main():
    args = parse_args()

    if args.okapi is not None:
        if args.raw is None:
            raise ValueError(
                "raw frequencies vector file required for computing okapi similarities"
            )
        sim_matrix = generate_similarity_matrix_okapi(args.okapi, args.raw)
        sim_matrix.insert(0, column="name", value=sim_matrix.index)
        sim_matrix.to_csv("similarities_okapi.csv", index=False)

    if args.tfidf is not None:
        sim_matrix = generate_similarity_matrix_tfidf(args.tfidf)
        sim_matrix.insert(0, column="name", value=sim_matrix.index)
        sim_matrix.to_csv("similarities_tfidf.csv", index=False)


if __name__ == "__main__":
    main()
