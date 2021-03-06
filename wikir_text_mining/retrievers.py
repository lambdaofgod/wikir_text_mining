import scipy
from sklearn import linear_model
import rank_bm25
import numpy as np
from typing import List
import attr
from sklearn import base, decomposition, feature_extraction, pipeline, metrics
import pandas as pd
from sklearn.base import TransformerMixin
from operator import itemgetter
import itertools
from wikir_text_mining import vectorizers
from warnings import simplefilter
import mlutil


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

    @classmethod
    def interpolate(cls, old_score, new_score, alpha=0.5):
        s_min, s_max = min(old_score), max(old_score)
        old_score = (old_score - s_min) / (s_max - s_min)

        s_min, s_max = min(new_score), max(new_score)
        new_score = (new_score - s_min) / (s_max - s_min)

        score = old_score * (1 - alpha) + new_score * alpha
        return score


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
        default=vectorizers.BM25Vectorizer()
    )
    clf: base.ClassifierMixin = attr.ib(
        default=linear_model.LogisticRegression(penalty='l1', solver='liblinear')
    )
    top_used = attr.ib(default=30)
    bottom_used = attr.ib(default=30)
    alpha = attr.ib(default=0.5)
    text_col = attr.ib(default='text')

    def retrieve(self, query, k=100, alpha=None):
        pseudo_relevant_df = self._retrieve_bm25(query, k)
        clf_scores = self._get_classifier_scores(pseudo_relevant_df)
        pseudo_relevant_df['classification_score'] = clf_scores
        pseudo_relevant_df['relevance_score'] = pseudo_relevant_df['score']
        pseudo_relevant_df['score'] = self.interpolate(
            pseudo_relevant_df['relevance_score'],
            pseudo_relevant_df['classification_score'],
            alpha=self.alpha if alpha is None else alpha
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

        score =  new_score * alpha + old_score * (1 - alpha)
        return score

    def _get_classifier_scores(self, pseudo_relevant_df):
        pseudo_relevant_texts = pseudo_relevant_df[self.text_col]
        pseudo_relevant_features = self.vectorizer.transform(pseudo_relevant_texts)
        positive_features = pseudo_relevant_features[:self.top_used]
        negative_features = pseudo_relevant_features[-self.bottom_used:]
        X_train = self._generic_vstack(positive_features, negative_features)
        y_train = np.ones(X_train.shape[0])
        y_train[self.top_used:] = 0
        self.clf.fit(X_train, y_train)
        return self.clf.predict_proba(pseudo_relevant_features)[:, 1]

    @classmethod
    def _generic_vstack(cls, m1, m2):
        if type(m1) == np.ndarray:
            return np.vstack([m1, m2])
        elif type(m1) == scipy.sparse.csr.csr_matrix:
            return scipy.sparse.vstack([m1, m2])
        else:
            raise ValueError('Can only stack ndarrays or sparse matrices')


class TopicModelRetriever(Retriever):

    def __init__(self,
             bm25,
             documents_df,
             vectorizer=vectorizers.BM25Vectorizer(nonnegative=True),
             topic_modeler=decomposition.NMF(n_components=10, max_iter=100, random_state=0),
             text_col='text',
             alpha=0.5,
             fit_transformers=True,
             features=None):
        self.bm25 = bm25
        self.documents_df = documents_df
        self._text_col = text_col
        self.feature_extractor = self._initialize_feature_extractor(
            vectorizer,
            topic_modeler,
            documents_df[text_col],
            fit_transformers)
        if features is None:
            self.features = self.feature_extractor.fit_transform(documents_df[text_col])
        else:
            self.features = features
        self.alpha = alpha

    def retrieve(self, query, k=100, alpha=None):
        pseudo_relevant_df = self._retrieve_bm25(query, k)
        reindexer = pd.DataFrame({'index': pd.RangeIndex(self.documents_df.shape[0])}, index=self.documents_df.index)
        indices = reindexer.loc[pseudo_relevant_df.index]['index']
        similarities = self.calculate_similarity(query, self.features[indices.values])
        pseudo_relevant_df['similarity_score'] = similarities
        pseudo_relevant_df['relevance_score'] = pseudo_relevant_df['score']
        pseudo_relevant_df['score'] = self.interpolate(
            pseudo_relevant_df['relevance_score'],
            pseudo_relevant_df['similarity_score'],
            alpha=self.alpha if alpha is None else alpha
        )
        return pseudo_relevant_df.sort_values(by='score', ascending=False)

    def _initialize_feature_extractor(self, vectorizer, topic_modeler, documents, fit_transformers):
        feature_extractor = pipeline.make_pipeline(vectorizer, topic_modeler)
        if fit_transformers:
            return feature_extractor.fit(documents)
        else:
            return feature_extractor

    def calculate_similarity(self, query, pseudo_relevant_features, method='cosine'):
        assert method in ['cosine']
        query_features = self.feature_extractor.transform([' '.join(query)])
        if method == 'cosine':
            return metrics.pairwise.cosine_similarity(query_features, pseudo_relevant_features).reshape(-1)


@attr.s
class QueryExpanderRetriever(Retriever):

    def retrieve(self, query, k=100, alpha=None):
        expanded_query = query + self.expand_query(query)
        return self._retrieve_bm25(expanded_query, k)


@attr.s
class WordEmbeddingQueryExpander(QueryExpanderRetriever):

    bm25: rank_bm25.BM25 = attr.ib()
    documents_df: pd.DataFrame = attr.ib()
    embedder: mlutil.embeddings.EmbeddingVectorizer = attr.ib()
    n_expanded_words = attr.ib(default=10)
    text_col = attr.ib(default='text')

    def expand_query(self, query, n_words=None):
        n_words = self.n_expanded_words if n_words is None else n_words
        vocab = self.embedder.word_embeddings.vocab.keys()
        similar_words_with_scores = list(
            itertools.chain.from_iterable(
                self.embedder.word_embeddings.most_similar_cosmul([word]) for word in query
                if word in vocab
            )
        )
        similar_words_by_closeness = sorted(similar_words_with_scores, key=itemgetter(1))
        similar_words = [w for w, __ in similar_words_by_closeness if w not in query]
        return similar_words[:n_words]