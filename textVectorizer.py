<<<<<<< Updated upstream
=======
from os import write
>>>>>>> Stashed changes
import pandas as pd
import numpy as np
import time
from scipy.sparse import csr_matrix
import re
from nltk.stem.porter import *
from collections import defaultdict
import os.path
import math
import argparse
<<<<<<< Updated upstream
=======
import multiprocessing as mp
>>>>>>> Stashed changes

STOPWORDS_PATH = './stopwords'
# replace all non-whitespace, digits, and other non-alphabetic characters
to_replace = re.compile(r'(?!\s)(\W|\d+)')
# split on any whitespace
whitespace_delimiters = re.compile('\s+')

# simple utility function to read stopwords from a list of stopwords files, 
# and combine them into one big set
def read_stopwords(files):
    stopwords = set()
    for words_file in files:
        with open(words_file) as f:
            for line in f.readlines():
                # strip out all the characters we don't want
                line = re.sub(to_replace, '', line)
                word = line.strip()
                if len(word) > 0:
                    stopwords.add(word.lower())
    
    return stopwords

# defines our set of stopwords to throw out
STOPWORDS = read_stopwords([os.path.join(STOPWORDS_PATH, p) for p in os.listdir(STOPWORDS_PATH)])

# Given a chunk of text, returns contiguous blocks of alphabetic characters (presumably, words)
def tokenize(text):
    stemmer = PorterStemmer()
    
    # strip off all the undesirable bits (punctuation, numbers, etc.)
    stripped = re.sub(to_replace, '', text)
    freqs = defaultdict(lambda: 0)
    for word in re.split(whitespace_delimiters, stripped):
        if word == '':
            continue
            
        # normalize all words to lowercase
        word = word.lower()
        # add stemmed word to frequency count if it is not a stopword
        if word not in STOPWORDS:
            stemmed = stemmer.stem(word)
            freqs[stemmed] += 1
            
        
    return freqs

# given the term frequencies for a single document (f_j), all document frequencies (d),
# the maximum frequencies for every keyword, and the positions word weights should be placed in,
# and the total number of documents,
# creates a vector which is the size of the vocabulary, and calculates the tf-idf weights
# for this particular set of frequencies.
def vectorize_document_tfidf(freqs, doc_freqs, max_freqs, positions, n):
    vec = np.zeros(len(positions))
    for keyword in freqs:
        tf_idf = (freqs[keyword] / max_freqs[keyword]) * math.log2(n/doc_freqs[keyword])
        vec[positions[keyword]] = tf_idf
    
    return vec


K1 = 1
B = 0.75
# given the term frequencies for a single document (f_j), all document frequencies (d),
# the maximum frequencies for every keyword, and the positions word weights should be placed in,
# and the total number of documents,
# creates a vector which is the size of the vocabulary, and calculates the okapi weights
# for this particular set of frequencies.
def vectorize_document_okapi(freqs, doc_freqs, max_freqs, positions, file_size, total_file_size, n):
    vec = np.zeros(len(positions))
    avg_doc_len = total_file_size / n

    for keyword in freqs:
        okapi = ((K1+1)*freqs[keyword])/(K1*(1-B+B*file_size/avg_doc_len) + freqs[keyword]) * math.log((n-doc_freqs[keyword]+0.5)/(doc_freqs[keyword]+0.5))
        vec[positions[keyword]] = okapi
    
    return vec

<<<<<<< Updated upstream
=======
# vectorizes a document using only raw frequency counts. This is useful for computing the okapi similarity
# between two vectors
def vectorize_document_raw_freq(freqs, positions):
    vec = np.zeros(len(positions))
    for keyword in freqs:
        vec[positions[keyword]] = freqs[keyword]

    return vec
>>>>>>> Stashed changes
# given doc_freqs, a dictionary of terms to the number of documents in which the terms occur,
# and name_to_freqs, a dict of dicts mapping of training file names to their respective word frequencies
# computes - for every keyword in doc_freqs - the maximum frequency of the word across all documents 
# we need this for normalizing the keyword frequencies when we vectorize
def max_term_frequencies(doc_freqs, name_to_freqs):
    max_freqs = {}
    for keyword in doc_freqs:
            # determine which document has the maximum frequency for a particular keyword
            max_freq_doc = max(name_to_freqs, 
                                    key=lambda doc_name: 0 if keyword not in name_to_freqs[doc_name] else 
                                        name_to_freqs[doc_name][keyword])
            max_freqs[keyword] = name_to_freqs[max_freq_doc][keyword]
    
    return max_freqs
            
def extract_words(data_file: str):
    with open(data_file, 'r') as f:
        return tokenize(f.read())

<<<<<<< Updated upstream
def vectorize_dataset(root_directory: str, tfidf_output_file, okapi_output_file):
=======
def write_to_csv(output_file: str, vector_rows: list[list[int]], name_to_freqs:  dict[str, dict[str, int]], doc_freqs: dict[str, int]):
    df = pd.DataFrame(vector_rows, index=name_to_freqs.keys(),columns=list(range(len(doc_freqs))))
    df.insert(0, column='file', value=df.index)
    df['file'] = df['file'].astype(pd.StringDtype())

    with open(output_file, 'w+') as f:
        df.to_csv(f, index=False)

def vectorize_tfidf(tfidf_output_file, name_to_freqs, doc_freqs, max_freqs, positions, total_documents):
    tfidf_rows = []
    for data_file in sorted(name_to_freqs):
        tfidf = vectorize_document_tfidf(
            name_to_freqs[data_file], 
            doc_freqs, 
            max_freqs, 
            positions, 
            total_documents)
        tfidf_rows.append(tfidf)
    write_to_csv(tfidf_output_file, tfidf_rows, name_to_freqs, doc_freqs)

def vectorize_okapi(okapi_output_file, name_to_freqs, doc_freqs, max_freqs, positions, file_sizes, total_file_size, total_documents):
    okapi_rows = []
    for data_file in sorted(name_to_freqs):
        tfidf = vectorize_document_okapi(
            name_to_freqs[data_file], 
            doc_freqs, 
            max_freqs, 
            positions, 
            file_sizes[data_file],
            total_file_size,
            total_documents)
        okapi_rows.append(tfidf)
    
    write_to_csv(okapi_output_file, okapi_rows, name_to_freqs, doc_freqs)

def vectorize_raw(raw_output_file, name_to_freqs, doc_freqs, positions):
    raw_rows = []
    for data_file in sorted(name_to_freqs):
        raw_vec = vectorize_document_raw_freq(name_to_freqs[data_file], positions)
        raw_rows.append(raw_vec)
    
    write_to_csv(raw_output_file, raw_rows, name_to_freqs, doc_freqs)

def vectorize_dataset(root_directory: str, tfidf_output_file=None, okapi_output_file=None, raw_output_file=None):
>>>>>>> Stashed changes
    # data file -> author mapping so we can create a ground truth file
    authors = {}
    # maps training file name -> document frequencies
    # essentially a dict of dict(keyword -> keyword count)
    name_to_freqs = {}
    file_sizes = {}
    doc_freqs = defaultdict(lambda: 0)
    total_documents = 0
    total_file_size = 0
    train_directory = os.path.join(root_directory, 'C50train')
    test_directory = os.path.join(root_directory, 'C50test')

    train_files = [os.path.join(train_directory, name) for name in os.listdir(train_directory)]
    test_files = [os.path.join(test_directory, name) for name in os.listdir(test_directory)]
    
    for path in train_files+test_files:
        for data_file in os.listdir(path):
            authors[data_file] = os.path.basename(path)
            file_sizes[data_file] = os.path.getsize(path)
            total_file_size += file_sizes[data_file]

            # determine the term frequency for all words in this data file
            freqs = extract_words(os.path.join(path, data_file))
            
            # ensure that the total document frequency is incremented by one
            # for each of the words we found in the document
            for word in freqs:
                doc_freqs[word] += 1
            
            # store the word frequencies for this particular datafile
            name_to_freqs[data_file] = freqs
            
            total_documents += 1
    
    # create ground truth df
<<<<<<< Updated upstream
    ground_truth = pd.DataFrame(index=sorted(name_to_freqs.keys()), columns=['file', 'author'])
    for (k, v) in authors.items():
        ground_truth.at[k, 'author'] = v
=======
    ground_truth_rows = []
    for (k, v) in authors.items():
        ground_truth_rows.append([k, v])
    ground_truth = pd.DataFrame(ground_truth_rows, index=sorted(name_to_freqs.keys()), columns=['file', 'author'])

    # filter out all terms that occur only in one document
    single_occurence_terms = [t for t, f in doc_freqs.items() if f == 1]
    for term in single_occurence_terms:
       del doc_freqs[term] 
       for (_, freqs) in name_to_freqs.items():
           if term in freqs:
               del freqs[term]
    
>>>>>>> Stashed changes
    
    # compute the maximum frequencies for every term
    max_freqs = max_term_frequencies(doc_freqs, name_to_freqs)
    
    # we are creating a vector which is the length of our vocabulary set,
    # so we must assign each word a unique 'dimension' in this vector
    positions = dict(zip(sorted(doc_freqs), range(len(doc_freqs))))
<<<<<<< Updated upstream
    
    tfidf_rows = []
    okapi_rows = []
    for data_file in name_to_freqs:
        tfidf = vectorize_document_tfidf(
            name_to_freqs[data_file], 
            doc_freqs, 
            max_freqs, 
            positions, 
            total_documents)
        tfidf_rows.append(tfidf)

        okapi = vectorize_document_okapi(
            name_to_freqs[data_file], 
            doc_freqs, 
            max_freqs, 
            positions, 
            file_sizes[data_file], 
            total_file_size,
            total_documents)
        
        okapi_rows.append(okapi)

    
    df_tfidf = pd.DataFrame(tfidf_rows, index=name_to_freqs.keys(),columns=list(range(len(doc_freqs))))
    with open(tfidf_output_file, 'w+') as f:
        df_tfidf.to_csv(f)
    
    df_okapi = pd.DataFrame(okapi_rows, index=name_to_freqs.keys(),columns=list(range(len(doc_freqs))))
    with open(okapi_output_file, 'w+') as f:
        df_okapi.to_csv(f)

    with open('ground_truth.csv', 'w+') as f:
        ground_truth.to_csv(f)
=======

    if tfidf_output_file is not None:
        vectorize_tfidf(tfidf_output_file, name_to_freqs, doc_freqs, max_freqs, positions, total_documents)
    
    if okapi_output_file is not None:
        vectorize_okapi(okapi_output_file, name_to_freqs, doc_freqs, max_freqs, positions, file_sizes, total_file_size, total_documents)
    
    if raw_output_file is not None:
        vectorize_raw(raw_output_file,name_to_freqs, doc_freqs, positions)

    if not os.path.exists('ground_truth.csv'):
        with open('ground_truth.csv', 'w+') as f:
            ground_truth.to_csv(f, index=False)
>>>>>>> Stashed changes

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset-root', type=str, required=True)
<<<<<<< Updated upstream
    parser.add_argument('-tfidf', '--output-file-tfidf', type=str, required=True)
    parser.add_argument('-okapi', '--output-file-okapi', type=str, required=True)
=======
    parser.add_argument('-tfidf', '--output-file-tfidf', type=str)
    parser.add_argument('-okapi', '--output-file-okapi', type=str)
    parser.add_argument('-raw', '--output-file-raw', type=str)
>>>>>>> Stashed changes

    return parser.parse_args()

def main():
    args = parse_args()
<<<<<<< Updated upstream
    vectorize_dataset(args.dataset_root, args.output_file_tfidf, args.output_file_okapi)
=======
    vectorize_dataset(args.dataset_root, args.output_file_tfidf, args.output_file_okapi, args.output_file_raw)
>>>>>>> Stashed changes

if __name__ == "__main__":
    main()