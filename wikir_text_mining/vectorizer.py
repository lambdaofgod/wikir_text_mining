# coding: UTF-8
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_extraction.text import _document_frequency, CountVectorizer
import tqdm
import funcy


class BM25Transformer(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    use_idf : boolean, optional (default=True)
    k1 : float, optional (default=2.0)
    b : float, optional (default=0.75)
    References
    ----------
    Okapi BM25: a non-binary model - Introduction to Information Retrieval
    http://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html
    """
    def __init__(self, use_idf=True, k1=2.0, b=0.75):
        self.use_idf = use_idf
        self.k1 = k1
        self.b = b

    def fit(self, X):
        """
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            document-term matrix
        """
        if not sp.issparse(X):
            X = sp.csc_matrix(X)
        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)
            idf = np.log((n_samples - df + 0.5) / (df + 0.5))
            self._idf_diag = sp.spdiags(idf, diags=0, m=n_features, n=n_features)
        return self

    def transform(self, X, copy=True):
        """
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            document-term matrix
        copy : boolean, optional (default=True)
        """
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            # preserve float family dtype
            X = sp.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)

        n_samples, n_features = X.shape

        # Document length (number of terms) in each row
        # Shape is (n_samples, 1)
        dl = X.sum(axis=1)
        # Number of non-zero elements in each row
        # Shape is (n_samples, )
        sz = X.indptr[1:] - X.indptr[0:-1]
        # In each row, repeat `dl` for `sz` times
        # Shape is (sum(sz), )
        # Example
        # -------
        # dl = [4, 5, 6]
        # sz = [1, 2, 3]
        # rep = [4, 5, 5, 6, 6, 6]
        rep = np.repeat(np.asarray(dl), sz)
        # Average document length
        # Scalar value
        avgdl = np.average(dl)
        # Compute BM25 score only for non-zero elements
        data = X.data * (self.k1 + 1) / (X.data + self.k1 * (1 - self.b + self.b * rep / avgdl))
        X = sp.csr_matrix((data, X.indices, X.indptr), shape=X.shape)

        if self.use_idf:
            check_is_fitted(self, '_idf_diag', 'idf vector is not fitted')

            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            # *= doesn't work
            X = X * self._idf_diag

        return X


class BM25Vectorizer(BaseEstimator, TransformerMixin):

    """
    Parameters
    ----------
    use_idf : boolean, optional (default=True)
    k1 : float, optional (default=2.0)
    b : float, optional (default=0.75)
    References
    ----------
    Okapi BM25: a non-binary model - Introduction to Information Retrieval
    http://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html
    """

    def __init__(self, use_idf=True, k1=2.0, b=0.75):
        self.count_vectorizer = CountVectorizer()
        self.bm25_transformer = BM25Transformer(use_idf, k1, b)

    def fit(self, X):
        document_term_matrix = self.count_vectorizer.fit_transform(X)
        self.bm25_transformer.fit(document_term_matrix)
        return self

    def transform(self, X, copy=True):
        document_term_matrix = self.count_vectorizer.transform(X)
        return self.bm25_transformer.transform(document_term_matrix, copy=copy)


class SelfAttentionVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, feature_pipeline):
        self.feature_pipeline = feature_pipeline

    def fit(self, X):
        return self

    def transform(self, X):
        return self._get_features(X)

    def _get_features(self, texts, max_min_features=False, chunk_size=10, verbose=False):
        text_features = []
        _iter = funcy.chunks(chunk_size, texts)
        if verbose:
            _iter = tqdm.tqdm(_iter, total=int(np.ceil(len(texts) / chunk_size)))
        for t in _iter:
            features = np.array(self.feature_pipeline.transform(t))
            feature_types = [features.mean(axis=1)]
            if max_min_features:
                feature_types = feature_types + [features.max(axis=1), features.min(axis=1)]
            concat_features = np.hstack(feature_types)
            text_features.append(concat_features)
        return np.vstack(text_features)