import pickle

import pandas as pd
from mlutil import embeddings
from sklearn import linear_model, naive_bayes, feature_extraction, decomposition

from wikir_text_mining import retrievers
from wikir_text_mining import vectorizers


relevance_model = pickle.load(open('artifacts/bm25.pkl', 'rb'))
documents_df = pd.read_csv('wikIR1k/documents_preprocessed.csv', index_col='id_right')


def predefined_retrievers(
        retriever_type,
        text_col,
        alpha,
        vectorizer='bm25',
        word_embedding_model='glove-wiki-gigaword-50'
    ):
    assert vectorizer in ['bm25', 'tfidf']
    if vectorizer == 'bm25':
        vectorizer = vectorizers.BM25Vectorizer()
    else:
        vectorizer = feature_extraction.text.TfidfVectorizer()
    if retriever_type == 'word_embedding_classifier_retriever':
        word_embedding_vectorizer = embeddings.AverageWordEmbeddingsVectorizer.from_gensim_embedding_model(word_embedding_model)
        return retrievers.ClassifierRetriever(
            bm25=relevance_model,
            documents_df=documents_df,
            vectorizer=word_embedding_vectorizer,
            clf=linear_model.LogisticRegression(solver='lbfgs'),
            alpha=alpha,
            text_col=text_col
        )
    elif retriever_type == 'classifier_retriever':
        clasifier_retriever = retrievers.ClassifierRetriever(
            bm25=relevance_model,
            documents_df=documents_df,
            vectorizer=vectorizer,
            clf=linear_model.LogisticRegression(penalty='l1', solver='liblinear'),
            alpha=alpha,
            text_col=text_col
        )
        clasifier_retriever.vectorizer.fit(documents_df[text_col])
        return clasifier_retriever
    elif retriever_type == 'naive_bayes_classifier_retriever':
        clasifier_retriever = retrievers.ClassifierRetriever(
            bm25=relevance_model,
            documents_df=documents_df,
            vectorizer=feature_extraction.text.CountVectorizer(),
            clf=naive_bayes.MultinomialNB(),
            alpha=alpha,
            text_col=text_col
        )
        clasifier_retriever.vectorizer.fit(documents_df[text_col])
        return clasifier_retriever
    elif retriever_type == 'topic_model_retriever':
        retriever = pickle.load(open('artifacts/topic_model_retriever.pkl', 'rb'))
        print(retriever)
        return retriever
    elif retriever_type == 'baseline':
        return retrievers.BM25Retriever(relevance_model, documents_df)
