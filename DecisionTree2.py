import pandas as pd
import math
import json
from collections import defaultdict
import enum
import numpy as np
import time

class SplitMetric(enum.Enum):
    INFO_GAIN = 0
    INFO_GAIN_RATIO =1

class ClassificationResult:
    def __init__(self, confusion_matrix):
        self.confusion_matrix = confusion_matrix
        self.update_metrics()
    
    def update_metrics(self):
        self.total_classified = self.calc_total_records_classified()
        self.total_correctly_classified = self.calc_total_correctly_classified()
        self.total_incorrectly_classified = self.calc_total_incorrectly_classified()
        self.overall_accuracy, self.overall_error = self.calc_overall_accuracy_error()
    
    def calc_total_records_classified(self):
        total = 0
        for col in self.confusion_matrix:
            total += self.confusion_matrix[col].sum()
        
        return total

    def calc_total_correctly_classified(self):
        return sum(self.confusion_matrix.at[row, col] 
            for (row, col) in 
            zip(self.confusion_matrix.index, self.confusion_matrix.columns))

    def calc_total_incorrectly_classified(self):
        return sum(self.confusion_matrix.at[row, col] 
            for row in self.confusion_matrix.index 
            for col in self.confusion_matrix.columns if row != col)
    
    def calc_overall_accuracy_error(self):
        acc = self.total_correctly_classified / self.total_classified
        return (acc, 1-acc)
    
    def combine(self, other):
        self.confusion_matrix = self.confusion_matrix.add(other.confusion_matrix, fill_value=0)
        self.update_metrics()
    
# Main decision tree classifier class. Contains logic for efficiently calculating
# entropy, as well as main tree-building procedure
class DecisionTreeClassifier:
    def __init__(self, dataset, class_col, attributes, gain_thresh=0.01, metric=SplitMetric.INFO_GAIN, tree=None):
        self.dataset = dataset
        self.attributes = attributes
        self.class_col = class_col
        self.class_col_domain = dataset[class_col].unique()
        self.gain_thresh = gain_thresh
        self.default_indent = 2
        self.metric = metric

        if tree is not None:
            self.tree = tree
        else:
            self.create_tree()



    def fast_entropy(self, class_column, class_domain):
        total_values = len(class_column)

        value_counts = class_column.value_counts()
        probs = np.array([0 if c not in value_counts
                        else value_counts[c]/total_values for c in class_domain])
        return -np.sum(np.multiply(probs, np.log2(probs, out=np.zeros(len(probs)), where=(probs!=0))))

    def find_best_split_fast(self, attr, data, class_col, domain):
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

        ent_minuses = np.array([self.fast_entropy(df[class_col], domain) for df in df_lt])
        ent_pluses = np.array([self.fast_entropy(df[class_col], domain) for df in df_gt])

        ents = np.add(np.multiply(weights_lt, ent_minuses), np.multiply(weights_gt, ent_pluses))
        p0 = np.full(len(ents), self.fast_entropy(sorted_data[class_col], domain))
        gains = np.add(p0,-ents)

    
        m_idx = np.argmax(gains) #np.where(gains == max_gain)[0][0]
        max_gain = np.max(gains)

        return (sorted_column.iloc[splt_idxs[m_idx]], max_gain)

    def select_splitting_attribute(self, attributes, data, class_col, domain, gain_thresh):
        gains = {}
        split_val = None
        for attr in sorted(attributes):
            (x, gain) =  self.find_best_split_fast(attr, data, class_col, domain)
            split_val = x
            gains[attr] = gain

        (max_gain_attr, max_gain) = max(gains.items(), key=lambda pair: pair[1])
        if max_gain >= gain_thresh:
            return (max_gain_attr, split_val)
    
        return None, None


    def find_most_frequent_label(self, dataset):
        counts = dataset[self.class_col].value_counts()[:1]
        return (counts.index[0], counts[counts.index[0]] / len(dataset)) 
    
    def create_tree_node(self, var, edges):
        return {"var": var, "edges": edges}
    
    def create_leaf_node(self, label, p):
        return {"decision": label, "p": p}
    
    def dump_to_json(self):
        return json.dumps(self.tree, indent=self.default_indent)

    def create_tree(self):
        root = {"dataset": 'tree.csv'}
        self.build_tree(self.dataset, set(self.attributes), root)
        self.tree = root

        return root
    
    def create_numeric_split_node(self, var, alpha):
        edge_val_left = {"value": alpha, "direction": "le"}
        edge_val_right = {"value": alpha, "direction": "gt"}
        return {
                "var": var,
                "edges": [{"edge": edge_val_left}, {"edge": edge_val_right}]
        }

    def create_ghost_node(self, dataset):
        (lbl, prob) =  self.find_most_frequent_label(dataset)
        return self.create_leaf_node(lbl, prob)

    def split(self, tree: dict, attributes: list[str], dataset: pd.DataFrame, splitting_attr: str, alpha: float):
        # if alpha is not none, we know this is a numeric attribute
        if alpha is not None:
            if len(dataset[splitting_attr].unique()) == 1:
                tree["leaf"] = self.create_ghost_node(dataset)
                return

            dataset_le = dataset[dataset[splitting_attr] <= alpha]
            dataset_gt = dataset[dataset[splitting_attr] > alpha]
            if len(dataset_le) == 0 or len(dataset_gt) == 0:
                tree["leaf"] = self.create_ghost_node(dataset)
                return
            
            tree["node"] = self.create_numeric_split_node(splitting_attr, alpha)
            self.build_tree(dataset_le, attributes, tree["node"]["edges"][0]['edge'])
            self.build_tree(dataset_gt, attributes, tree["node"]["edges"][1]['edge'])
            return

    def build_tree(self, dataset: pd.DataFrame, attributes: set[str], tree: dict):
        class_values = dataset[self.class_col].unique()
        if len(class_values) == 1:
            tree["leaf"] = self.create_leaf_node(class_values[0], 1.0)
        elif len(attributes) < 1:
            (lbl, prob) =  self.find_most_frequent_label(dataset)
            tree["leaf"] = self.create_leaf_node(lbl, prob)
        else:
            res = self.select_splitting_attribute(attributes, dataset, self.class_col, self.class_col_domain, self.gain_thresh)
            if res[0] is None and res[1] is None:
                tree["leaf"] = self.create_ghost_node(dataset)
                return
            
            (splitting_attr, alpha) = res
            self.split(tree, attributes, dataset, splitting_attr, alpha)

    def classify_datapoint(self, row):
        return self._classify_datapoint(row, self.tree)

    def _classify_datapoint(self, row, tree):
        if 'leaf' in tree:
            return tree['leaf']['decision']        

        current_var = tree['node']['var']
        v = row[current_var]
        edge_le, edge_gt = tree['node']['edges'][0]['edge'], tree['node']['edges'][1]['edge']
        if v <= edge_le['value']:
            return self._classify_datapoint(row, edge_le)
        else:
            return self._classify_datapoint(row, edge_gt)