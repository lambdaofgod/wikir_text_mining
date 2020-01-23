import tqdm
import nltk
import pandas as pd
import pytrec_eval
import numpy as np
from collections import OrderedDict


def make_relevance_dict(queries_df, relevant_indices):
    return {
        str(l): {
            str(k): v
            for (k, v) in relevant_indices[i].to_dict().items()
        }
        for i, l in enumerate(queries_df.index)
    }


def get_predicted_relevant_indices(queries_df, documents_df, bm25, k=100, verbose=True):
    bm25_relevant_indices = []
    queries_iter = queries_df.iterrows()
    if verbose:
        queries_iter = tqdm.tqdm(queries_iter, total=len(queries_df))
    for __, q in queries_iter:
        query = [nltk.stem.PorterStemmer().stem(t) for t in q['text_left'].split()]
        scores = bm25.get_scores(query)
        sorted_indices = np.argsort(scores)[::-1][:k]
        sorted_scores = scores[sorted_indices]
        document_indices = documents_df.index[sorted_indices]
        bm25_relevant_indices.append(pd.Series(sorted_scores, index=document_indices))
    return bm25_relevant_indices


def setup_evaluator_from_relevance_file(qrel_path, measures={"map","ndcg_cut","recall","P"}):
    with open(qrel_path, 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)

    return pytrec_eval.RelevanceEvaluator(qrel,measures)


def get_evaluation_df(predicted_relevance, evaluator):
    results = evaluator.evaluate(predicted_relevance)
    return pd.DataFrame.from_records(OrderedDict(results)).T


def make_score_dict(results_df, score_col='score_bm25'):
    return {
        str(row.name): row[score_col]
        for _, row in results_df.iterrows()
    }


def evaluate_query(query_id, query, retriever, evaluator, score_col='score', **retriever_kwargs):
    results_df = retriever.retrieve(query)
    results_dict = {str(query_id): make_score_dict(results_df, score_col=score_col)}
    evaluation_df = get_evaluation_df(results_dict, evaluator)
    return evaluation_df
