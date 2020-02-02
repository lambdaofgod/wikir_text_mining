import pickle

import pandas as pd
from mlutil import embeddings
from sklearn import linear_model, naive_bayes, feature_extraction, decomposition, neighbors

from wikir_text_mining import retrievers
from wikir_text_mining import vectorizers


relevance_model = pickle.load(open('artifacts/bm25.pkl', 'rb'))
documents_df = pd.read_csv('wikIR1k/documents_preprocessed.csv', index_col='id_right')


def predefined_retrievers(
        retriever_type,
        text_col,
        alpha,
        vectorizer='bm25',
        word_embedding_model='glove-wiki-gigaword-50',
        query_expander_n_expanded_words=10,
        topic_modeler_n_components=50,
        word_embedding_classifier='linear'
    ):
    assert vectorizer in ['bm25', 'tfidf']
    if retriever_type == 'word_embedding_classifier_retriever':
        word_embedding_vectorizer = embeddings.AverageWordEmbeddingsVectorizer.from_gensim_embedding_model(word_embedding_model)
        if word_embedding_classifier == 'linear':
            clf = linear_model.LogisticRegression(solver='lbfgs')
        elif word_embedding_classifier == 'knn':
            clf = neighbors.KNeighborsClassifier(n_neighbors=1, metric='cosine')
        return retrievers.ClassifierRetriever(
            bm25=relevance_model,
            documents_df=documents_df,
            vectorizer=word_embedding_vectorizer,
            clf=clf,
            alpha=alpha,
            text_col=text_col
        )
    elif retriever_type == 'classifier_retriever':
        if vectorizer == 'bm25':
            vectorizer = vectorizers.BM25Vectorizer()
        else:
            vectorizer = feature_extraction.text.TfidfVectorizer()
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
        if vectorizer == 'bm25':
            vectorizer = vectorizers.BM25Vectorizer(nonnegative=True)
        else:
            vectorizer = feature_extraction.text.TfidfVectorizer()
        retriever = retrievers.TopicModelRetriever(
            bm25=relevance_model,
            documents_df=documents_df,
            vectorizer=vectorizer,
            topic_modeler=decomposition.NMF(n_components=topic_modeler_n_components),
            text_col=text_col
        )
        print(retriever)
        return retriever
    elif retriever_type == 'baseline':
        return retrievers.BM25Retriever(relevance_model, documents_df)
    elif retriever_type == 'query_expander':
        word_embedding_vectorizer = embeddings.AverageWordEmbeddingsVectorizer.from_gensim_embedding_model(word_embedding_model)
        return retrievers.WordEmbeddingQueryExpander(
            bm25=relevance_model,
            documents_df=documents_df,
            embedder=word_embedding_vectorizer,
            n_expanded_words=query_expander_n_expanded_words
        )
