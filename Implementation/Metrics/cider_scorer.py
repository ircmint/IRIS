"""
cider_scorer.py

Minimal, self-contained CIDEr-D implementation (Vedantam et al. 2015).
Written from scratch because pycocoevalcap is not installed anywhere in
this environment (checked locally and on Ada) and installing it pulls in
a large, old dependency tree for one metric. This implements the same
TF-IDF-weighted n-gram (n=1..4) cosine-similarity formula CIDEr-D uses,
including the length-penalty Gaussian term -- it is not a stub.

Usage:
    from cider_scorer import compute_cider
    score = compute_cider(candidates: list[str], references: list[list[str]])
"""

import math
from collections import Counter, defaultdict

N_GRAM_MAX = 4
SIGMA = 6.0


def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _counts_by_n(tokens):
    return {n: Counter(_ngrams(tokens, n)) for n in range(1, N_GRAM_MAX + 1)}


def compute_cider(candidates: list, references: list) -> float:
    """
    candidates: list of candidate strings, one per example.
    references: list of lists of reference strings, one list per example
                (CIDEr supports multiple references per example; pass a
                single-element list if you only have one gold text).
    Returns the corpus-level mean CIDEr-D score.
    """
    assert len(candidates) == len(references)
    cand_tokens = [c.lower().split() for c in candidates]
    ref_tokens = [[r.lower().split() for r in refs] for refs in references]

    doc_freq = defaultdict(int)  # how many documents (both cand+refs) contain this n-gram
    all_docs = []
    for toks in cand_tokens:
        all_docs.append(_counts_by_n(toks))
    for refs in ref_tokens:
        for r in refs:
            all_docs.append(_counts_by_n(r))
    num_docs = len(all_docs)
    for doc in all_docs:
        for n in range(1, N_GRAM_MAX + 1):
            for ngram in doc[n]:
                doc_freq[(n, ngram)] += 1

    def tfidf_vec(counts_by_n, doc_len):
        vec = {}
        for n in range(1, N_GRAM_MAX + 1):
            for ngram, count in counts_by_n[n].items():
                df = doc_freq.get((n, ngram), 1)
                idf = math.log(max(1.0, num_docs / df))
                tf = count / max(1, doc_len)
                vec[(n, ngram)] = tf * idf
        return vec

    def cos_sim(v1, v2, len1, len2, n):
        keys = set(k for k in v1 if k[0] == n) | set(k for k in v2 if k[0] == n)
        dot = sum(v1.get(k, 0.0) * v2.get(k, 0.0) for k in keys)
        norm1 = math.sqrt(sum(v1.get(k, 0.0) ** 2 for k in keys)) or 1e-12
        norm2 = math.sqrt(sum(v2.get(k, 0.0) ** 2 for k in keys)) or 1e-12
        gauss = math.exp(-((len1 - len2) ** 2) / (2 * SIGMA ** 2))
        return (dot / (norm1 * norm2)) * gauss

    scores = []
    for cand_c, refs_c, cand_toks, refs_toks in zip(
            [_counts_by_n(t) for t in cand_tokens], ref_tokens, cand_tokens, ref_tokens):
        cand_vec = tfidf_vec(cand_c, len(cand_toks))
        per_ref_scores = []
        for ref_t in refs_toks:
            ref_c = _counts_by_n(ref_t)
            ref_vec = tfidf_vec(ref_c, len(ref_t))
            n_scores = [cos_sim(cand_vec, ref_vec, len(cand_toks), len(ref_t), n)
                        for n in range(1, N_GRAM_MAX + 1)]
            per_ref_scores.append(sum(n_scores) / N_GRAM_MAX)
        scores.append((sum(per_ref_scores) / len(per_ref_scores)) * 10.0 if per_ref_scores else 0.0)

    return sum(scores) / len(scores) if scores else 0.0
