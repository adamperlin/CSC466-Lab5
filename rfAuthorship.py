import pandas as pd
import numpy as np
import argparse
import random
from statistics import mode
from DecisionTree import Node, Edge, Leaf
import time


def write_tree_to_json(T):
    data = {}
    if type(T) == Node:
        data["node"] = {"var": T.var, "edges": []}
        for e in T.edges:
            data["node"]["edges"].append(write_tree_to_json(e))
    elif type(T) == Leaf:
        data["leaf"] = {"decision": T.decision, "p": T.p}
    elif type(T) == Edge:
        try:
            key = "node"
            val = write_tree_to_json(T.node)["node"]
        except KeyError:
            key = "leaf"
            val = write_tree_to_json(T.node)["leaf"]
        vals = T.value.split(" ")
        data["edge"] = {
            "value": vals[1],
            "direction": vals[0],
            key: val,
        }

    return data

def recurse_tree(tree, row):
    try:
        leaf = tree["leaf"]
        return leaf
    except KeyError:
        node = tree["node"]
        attr = node["var"]
        class_val = row[attr].values[0]
        edges = node["edges"]
        for e in edges:
            direction = e["edge"]["direction"]
            if direction == "le":
                if str(class_val) <= str(e["edge"]["value"]):
                    return recurse_tree(e["edge"], row)
            else:
                if str(class_val) > str(e["edge"]["value"]):
                    return recurse_tree(e["edge"], row)


def get_m_attributes(attributes, m):
    # randomly select m attributes from dataframe without replacement
    return random.sample(attributes, m)

def construct_author_df(df, k):
    # shuffles the dataframe and returns a random subset of 50 unique author vectors
    authors_50 = []
    for author in df['author'].unique():
        d_author = df[df['author'] == author].sample(n=20, replace=True)
        authors_50.append(d_author)
    
    # choose k - 50 data points from the dataframe with replacement
    extra_df = df.sample(n=(k-1000), replace=True)
    k_df = pd.concat(authors_50 + [extra_df])
    return k_df

def fast_entropy(class_column, class_domain):
    total_values = len(class_column)

    value_counts = class_column.value_counts()
    probs = np.array([0 if c not in value_counts
                    else value_counts[c]/total_values for c in class_domain.index])
    return -np.sum(np.multiply(probs, np.log2(probs, out=np.zeros(len(probs)), where=(probs!=0))))

def find_best_split_fast(attr, data, class_col, domain):
    sorted_data = data.filter(items=[attr, class_col]).sort_values(by=attr)

    total_len = len(data)
    sorted_column = sorted_data[attr]
    
    arr = np.array(sorted_column)
    
    # compute the last index of every unique value in the array
    #i.e., splt_idxs for the column [1, 1, 2, 2, 2, 3, 4, 5] would 
    # return [1, 4, ]
    #splt_idxs = np.unique([np.argwhere(arr == val)[-1] for val in arr])
    #splt_idxs = np.unique([np.argwhere(arr == val)[-1] for val in arr])
    splt_idxs = np.concatenate(np.array([np.argwhere(arr == val)[-1] for val in np.unique(arr)]))

    df_lt = [sorted_data.iloc[:idx] for idx in splt_idxs]
    df_gt = [sorted_data.iloc[idx:] for idx in splt_idxs]

    weights_lt = [0 if total_len == 0 else len(df)/total_len for df in df_lt]
    weights_gt = [0 if total_len == 0 else len(df)/total_len for df in df_gt]

    ent_minuses = np.array([fast_entropy(df[class_col], domain) for df in df_lt])
    ent_pluses = np.array([fast_entropy(df[class_col], domain) for df in df_gt])

    ents = np.add(np.multiply(weights_lt, ent_minuses), np.multiply(weights_gt, ent_pluses))
    p0 = np.full(len(ents), fast_entropy(sorted_data[class_col], domain))
    gains = np.add(p0,-ents)

   
    m_idx = np.argmax(gains) #np.where(gains == max_gain)[0][0]
    max_gain = np.max(gains)

    return (sorted_column.iloc[splt_idxs[m_idx]], max_gain)


def select_splitting_attribute(attributes, data, class_col, domain, gain_thresh):
    gains = {}
    split_val = None
    for attr in sorted(attributes):
        (x, gain) =  find_best_split_fast(attr, data, class_col, domain)
        split_val = x
        gains[attr] = gain

    (max_gain_attr, max_gain) = max(gains.items(), key=lambda pair: pair[1])
    if max_gain >= gain_thresh:
        return (max_gain_attr, split_val)
 
    return None, None

def C45(D, A, class_label, threshold=0.01):
    size = len(D)
    domain = D[class_label].value_counts(sort=True)
    # c = list(domain.items())[0]
    plurality = domain.index[0]
    plurality_size = domain.iloc[0]
    if len(domain) == 1:
        T = Leaf(plurality, 1.0)
    elif A == []:
        T = Leaf(plurality, plurality_size / size)
    else:
        splitting_attr, split = select_splitting_attribute(A, D, class_label, domain, threshold)

        if splitting_attr is None:
            T = Leaf(plurality, plurality_size / size)
        else:
            D_v = D.loc[D[splitting_attr] <= split]
            if len(D_v) != size and len(D_v) != 0:
                T = Node()
                T.var = splitting_attr
                T_v = C45(D_v, A, class_label, threshold)
                e = Edge()
                e.value = "le " + str(split)
                e.node = T_v
                T.edges.append(e)

                D_v = D.loc[D[splitting_attr] > split]
                T_v = C45(D_v, A, class_label, threshold)
                e = Edge()
                e.value = "gt " + str(split)
                e.node = T_v
                T.edges.append(e)
            else:
                T = Leaf(plurality_size, plurality_size / size)
    return T

def make_prediction(forest, tfidf, file_name):
    pred = []
    file_vector = tfidf[tfidf['file'] == file_name]
    for T in forest:
        tree = write_tree_to_json(T)
        author = recurse_tree(tree, file_vector)['decision']
        pred.append(author)

    return mode(pred)

def random_forest(rf_df, m, k, N, threshold):
    forest = []
    attributes = rf_df.columns.values.tolist()
    attributes.remove('author')
    for i in range(N):
        k_df = construct_author_df(rf_df, k)
        m_attributes = get_m_attributes(attributes, m)
        T = C45(k_df, m_attributes, 'author') 
        forest.append(T)
        print(f"[progress] constructed {i}/{N} trees")

    return forest

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-tfidf_file", "--tfidf_file", required=True)
    parser.add_argument("-ground_truth_file", "--ground-truth-file", required=True)
    parser.add_argument('-N', '--num-decision-trees', type=int, required=True)
    parser.add_argument('-m', '--num-attributes', type=int, required=True)
    parser.add_argument('-k', '--num-data-points', type=int, required=True)
    parser.add_argument('-thresh', '--threshold', type=int)

    return parser.parse_args()

def main():
    args = parse_args()
    tfidf = pd.read_csv(args.tfidf_file)
    ground_truth = pd.read_csv(args.ground_truth_file)

    rf_df = tfidf.merge(ground_truth, on='file', how='inner')
    rf_df.set_index('file', inplace=True)

    ground_truth.set_index('file', inplace=True)

    forest = random_forest(rf_df, args.num_attributes, args.num_data_points, args.num_decision_trees, args.threshold)
    
    predictions = pd.DataFrame(index=ground_truth.index, columns=['predicted_author'])
    for file_name in rf_df.index:
        predicted_author = make_prediction(forest, tfidf, file_name)
        predictions.at[file_name, 'predicted_author'] = predicted_author
    
    with open('predictions.csv', 'w+') as f:
        predictions.insert(0, column='file', value=predictions.index)
        predictions.to_csv(f, index=False)

if __name__ == "__main__":
    main()
