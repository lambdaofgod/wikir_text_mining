import scipy
from sklearn import linear_model
import rank_bm25
import numpy as np
from typing import List
import attr
from sklearn import base
import pandas as pd
from wikir_text_mining import vectorizer


class Retriever:

    def retrieve(self, query: List[str], k=100):
        raise NotImplementedError()

    def _retrieve_bm25(self, query: List[str], k=100):
        scores = self.bm25.get_scores(query)
        sorted_indices = np.argsort(scores)[::-1][:k]
        sorted_scores = scores[sorted_indices]
        results_df = self.documents_df.loc[sorted_indices]
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

    def retrieval_results(self, query, k=100):
        pseudo_relevant_df = self._retrieve_bm25(query, k)
        pseudo_relevant_texts = pseudo_relevant_df['text']
        pseudo_relevant_features = self.vectorizer.transform(pseudo_relevant_texts)
        positive_features = pseudo_relevant_features[:self.top_used]
        negative_features = pseudo_relevant_features[-self.bottom_used:]
        X_train = scipy.sparse.vstack([positive_features, negative_features])
        y_train = np.ones(X_train.shape[0])
        y_train[self.top_used:] = 0
        self.clf.fit(X_train, y_train)
        print(self.clf.score(X_train, y_train))
        clf_scores = self.clf.predict_proba(pseudo_relevant_features)[:,1]
        sorted_indices = np.argsort(clf_scores)[::-1][:k]
        sorted_scores = clf_scores[sorted_indices]
        results_df = pseudo_relevant_df.iloc[sorted_indices]
        results_df['score'] = sorted_scores
        return results_df, pseudo_relevant_df

    def fit_vectorizer(self, texts):
        self.vectorizer.fit(texts)
