import scipy
from sklearn import linear_model
import rank_bm25
import numpy as np
from typing import List
import attr
from sklearn import base
import pandas as pd
from wikir_text_mining import vectorizer
from warnings import simplefilter


simplefilter(action='ignore', category=FutureWarning)


class Retriever:

    def retrieve(self, query: List[str], k=100):
        raise NotImplementedError()

    def _retrieve_bm25(self, query: List[str], k=100):
        scores = self.bm25.get_scores(query)
        sorted_indices = np.argsort(scores)[::-1][:k]
        document_indices = self.documents_df.index[sorted_indices]
        sorted_scores = scores[sorted_indices]
        results_df = self.documents_df.loc[document_indices]
        results_df['score'] = sorted_scores
        return results_df


@attr.s
class BM25Retriever(Retriever):

    bm25: rank_bm25.BM25 = attr.ib()
    documents_df: pd.DataFrame = attr.ib()

    def retrieve(self, query, k=100):
        return self._retrieve_bm25(query, k)


@attr.s
class ClassifierRetriever(Retriever):
    """
    Uses classification of top and bottom documents from relevance list for reranking
    Idea from "The Simplest Thing That Can Possibly Work:Pseudo-Relevance Feedback Using Text Classification"

    Parameters:
        bm25: rank_bm25 model for relevance ranking
        documents_df: documents for retrieval
        vectorizer: transformer used for feature extraction for classifier, see wikir_text_mining.vectorizer
        clf: sklearn-compatible classifier
    """

    bm25: rank_bm25.BM25 = attr.ib()
    documents_df: pd.DataFrame = attr.ib()
    vectorizer: base.TransformerMixin = attr.ib(
        default=vectorizer.BM25Vectorizer()
    )
    clf: base.ClassifierMixin = attr.ib(
        default=linear_model.LogisticRegression(penalty='l1', solver='liblinear')
    )
    top_used = attr.ib(default=30)
    bottom_used = attr.ib(default=30)
    alpha = attr.ib(default=0.5)

    def retrieve(self, query, k=100):
        pseudo_relevant_df = self._retrieve_bm25(query, k)
        clf_scores = self._get_classifier_scores(pseudo_relevant_df)
        pseudo_relevant_df['classification_score'] = clf_scores
        pseudo_relevant_df['relevance_score'] = pseudo_relevant_df['score']
        pseudo_relevant_df['score'] = self.interpolate(
            pseudo_relevant_df['classification_score'],
            pseudo_relevant_df['relevance_score'],
            alpha=self.alpha
        )
        return pseudo_relevant_df.sort_values(by='score', ascending=False)

    def fit_vectorizer(self, texts):
        self.vectorizer.fit(texts)

    @classmethod
    def interpolate(cls, old_score, new_score, alpha=0.5):
        s_min, s_max = min(old_score), max(old_score)
        old_score = (old_score - s_min) / (s_max - s_min)

        s_min, s_max = min(new_score), max(new_score)
        new_score = (new_score - s_min) / (s_max - s_min)

        score = old_score * (1 - alpha) + new_score * alpha
        return score

    def _get_classifier_scores(self, pseudo_relevant_df, text_col='text'):
        pseudo_relevant_texts = pseudo_relevant_df[text_col]
        pseudo_relevant_features = self.vectorizer.transform(pseudo_relevant_texts)
        positive_features = pseudo_relevant_features[:self.top_used]
        negative_features = pseudo_relevant_features[-self.bottom_used:]
        X_train = self._generic_vstack(positive_features, negative_features)
        y_train = np.ones(X_train.shape[0])
        y_train[self.top_used:] = 0
        self.clf.fit(X_train, y_train)
        return self.clf.predict_proba(pseudo_relevant_features)[:,1]

    @classmethod
    def _generic_vstack(cls, m1, m2):
        if type(m1) == np.ndarray:
            return np.vstack([m1, m2])
        elif type(m1) == scipy.sparse.csr.csr_matrix:
            return scipy.sparse.vstack([m1, m2])
        else:
            raise ValueError('Can only stack ndarrays or sparse matrices')