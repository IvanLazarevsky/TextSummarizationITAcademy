import sys
from typing import List

from pymystem3 import Mystem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise
from tqdm import tqdm
import razdel
import numpy as np
import networkx as nx

def get_lemmatized_sentences(texts):
    mystem = Mystem(entire_input=False)
    result = []
    original_sentences = []
    for text in tqdm(texts):
        text_repr = []
        original_repr = []
        for sent in razdel.sentenize(text):
            lemmas = mystem.lemmatize(sent.text)
            text_repr.append(lemmas)
            original_repr.append(sent.text)

        result.append(text_repr)
        original_sentences.append(original_repr)
    return result, original_sentences

def train_tfidf(concatenated_sentences: List[List[str]]):
    vectorizer = TfidfVectorizer(preprocessor=lambda s:s, tokenizer=lambda s:s,lowercase=False,max_df=0.4,min_df=5,max_features=10_000)
    return vectorizer.fit(concatenated_sentences)

def compute_tfidf_matrix(sentences: List[List[str]], vectorizer):
    mtr = vectorizer.transform(sentences)
    return mtr

def compute_similarity(dense_or_sparse):
    return pairwise.cosine_similarity(dense_or_sparse)

def compute_tfidf_similarity_matrix(sentences: List[List[str]], vectorizer):
    return compute_similarity(compute_tfidf_matrix(sentences, vectorizer))


def get_pagerank_matrix(similarity_matrix, d, zero_diag=False, threshold=None):
    N = similarity_matrix.shape[0]
    similarity_matrix = similarity_matrix.copy()
    if threshold is not None:
        similarity_matrix = (similarity_matrix >= threshold).astype(np.float64)
    if zero_diag:
        np.fill_diagonal(similarity_matrix, 0)

    sims = similarity_matrix / similarity_matrix.sum(axis=1, keepdims=True)
    sims = np.nan_to_num(sims)

    sims = d * sims + np.ones((N, N)) * ((1 - d) / N)
    sims = sims / sims.sum(axis=1, keepdims=True)
    return sims

def draw_sentence_graph(m, sentences=None):
    g = nx.to_networkx_graph(m)
    weights = [g[u][v]['weight'] * 3 for u,v in g.edges()]
    if sentences is not None:
        labels = {i: s for i, s in zip(g.nodes(), sentences)}
    else:
        labels = {i:i+1 for i in g.nodes()}
    return nx.draw(g, labels=labels, with_labels=True, width=weights)


def power_iter(M, max_iterations, eps_norm):
    N = M.shape[0]
    v = np.ones(N) / N

    for i in range(max_iterations):
        v_next = M.T.dot(v)
        if np.linalg.norm(v_next - v) < eps_norm:
            return v_next
        v = v_next

    sys.stderr.write(f"Didn't converge after {max_iterations} iterations")

    return v