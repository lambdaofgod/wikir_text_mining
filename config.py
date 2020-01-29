import pickle

import pandas as pd
from mlutil import embeddings
from sklearn import linear_model, naive_bayes, feature_extraction

from wikir_text_mining import retrievers
from wikir_text_mining import vectorizers

relevance_model = pickle.load(open('artifacts/bm25.pkl', 'rb'))
documents_df = pd.read_csv('wikIR1k/documents.csv', index_col='id_right')
documents_df['text'] = documents_df['text_right']


def predefined_retrievers(retriever_type, classifier_retriever_alpha=0.5):
    if retriever_type == 'word2vec_classifier_retriever':
        word2vec_vectorizer = embeddings.AverageWordEmbeddingsVectorizer.from_gensim_embedding_model()
        return retrievers.ClassifierRetriever(
            bm25=relevance_model,
            documents_df=documents_df,
            vectorizer=word2vec_vectorizer,
            clf=linear_model.LogisticRegression(solver='lbfgs'),
            alpha=classifier_retriever_alpha
        )
    elif retriever_type == 'classifier_retriever':
        clasifier_retriever = retrievers.ClassifierRetriever(
            bm25=relevance_model,
            documents_df=documents_df,
            vectorizer=vectorizers.BM25Vectorizer(),
            clf=linear_model.LogisticRegression(penalty='l1', solver='liblinear'),
            alpha=classifier_retriever_alpha
        )
        clasifier_retriever.vectorizer.fit(documents_df['text'])
        return clasifier_retriever
    elif retriever_type == 'naive_bayes_classifier_retriever':
        clasifier_retriever = retrievers.ClassifierRetriever(
            bm25=relevance_model,
            documents_df=documents_df,
            vectorizer=feature_extraction.text.CountVectorizer(),
            clf=naive_bayes.MultinomialNB(),
            alpha=classifier_retriever_alpha
        )
        clasifier_retriever.vectorizer.fit(documents_df['text'])
        return clasifier_retriever
    elif retriever_type == 'topic_model_retriever':
        pass
    elif retriever_type == 'baseline':
        return retrievers.BM25Retriever(relevance_model, documents_df)
