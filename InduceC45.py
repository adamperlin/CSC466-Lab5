import pandas as pd
import sys
import math
import json
from DecisionTree import Node, Edge, Leaf


def main():
    pass

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
        if T.value.startswith("!"):
            vals = T.value.split(" ")
            data["edge"] = {
                "value": vals[1],
                "direction": vals[0].replace("!", ""),
                key: val,
            }
        else:
            data["edge"] = {"value": T.value, key: val}

    return data


def C45(D, A, threshold, class_label):
    # Check stopping conditions
    size = len(D)
    dom_class_label = dict(D[class_label].value_counts(sort=True))
    c = list(dom_class_label.items())[0]
    if len(dom_class_label) == 1:
        T = Leaf(c[0], 1.0)
    elif A == {}:
        T = Leaf(c[0], c[1] / size)
    else:
        splitting_attr, split = select_splitting_attribute(A, D, threshold, class_label)
        if splitting_attr is None:
            T = Leaf(c[0], c[1] / size)
        else:
            T = Node()
            T.var = splitting_attr
            dom = A[splitting_attr]
            D_v = D.loc[D[splitting_attr] <= split]
            if D_v.shape[0] != D.shape[0]:
                T_v = C45(D_v, A, threshold, class_label)
                e = Edge()
                e.value = "!le " + str(split)
                e.node = T_v
                T.edges.append(e)

                D_v = D.loc[D[splitting_attr] > split]
                T_v = C45(D_v, A, threshold, class_label)
                e = Edge()
                e.value = "!gt " + str(split)
                e.node = T_v
                T.edges.append(e)
    return T


def select_splitting_attribute(A, D, threshold, class_label):
    p0 = class_entropy(class_label, D)
    p = {}
    gain = {}
    for attr in A:
        x, entropies = find_best_split(p0, attr, D, class_label)
        p[attr] = entropy_alpha(attr, D, x, entropies)
        A[attr] = ("numeric", x)
        gain[attr] = p0 - p[attr]

    if gain == {}:
        print("attribute", A)

    best_attr = max(gain, key=gain.get)
    if gain[best_attr] > threshold:
        return best_attr, A[best_attr][1]
    return None, None


def find_best_split(p0, attr, D, class_label):
    dom = domain(class_label, D)
    gain = {}
    entropies = {}
    D = D.sort_values(by=[attr])
    for l in range(len(D) - 1):
        d = D.iloc[l]
        alpha = d[attr]
        # make sure alpha doesnt already exist in gain
        try:
            gain[alpha]
        except KeyError:
            D_alpha = D.loc[D[attr] <= alpha]
            cts_left = [len(D_alpha.loc[D_alpha[class_label] == i]) for i in dom]
            intermediate_cts_right = [dom[i] for i in dom]
            cts_right = [
                intermediate_cts_right[i] - cts_left[i] for i in range(len(cts_left))
            ]
            entropy_left = entropy_cont(cts_left)
            entropy_right = entropy_cont(cts_right)

            gain[alpha] = p0 - (entropy_left + entropy_right)
            entropies[alpha] = (entropy_left, entropy_right)

    best = max(gain, key=gain.get)
    return best, entropies[best]


def entropy_alpha(attr, D, x, entropies):
    size = len(D)
    left = len(D.loc[D[attr] <= x])
    right = size - left
    leftEntropy = entropies[0]
    rightEntropy = entropies[1]

    return ((left / size) * leftEntropy) + ((right / size) * rightEntropy)


def entropy_cont(counts):
    total = sum(counts)
    sum_entr = [(i / total) * math.log(i / total, 2) if i != 0 else 0 for i in counts]
    return -sum(sum_entr)


def class_entropy(class_label, D):
    size = D.shape[0]
    domain_vals = domain(class_label, D)

    entropy = -sum(
        [
            domain_vals[k] / size * math.log(domain_vals[k] / size, 2)
            for k in domain_vals
        ]
    )
    return entropy


def domain(attr, D):
    # sort by alphabetical
    return dict(D[attr].value_counts().sort_index(ascending=True))


if __name__ == "__main__":
    main()
