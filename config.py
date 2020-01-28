from wikir_text_mining import vectorizer, retriever
from sklearn import linear_model
from mlutil import embeddings
from wikir_text_mining import evaluation, retriever
import pickle
import pandas as pd


bm25 = pickle.load(open('bm25.pkl', 'rb'))
documents_df = pd.read_csv('wikIR1k/documents.csv', index_col='id_right')
documents_df['text'] = documents_df['text_right']

word2vec_vectorizer = embeddings.AverageWordEmbeddingsVectorizer.from_gensim_embedding_model()


PREDEFINED_RETRIEVERS = {
    'word2vec_classifier_retriever':
        retriever.ClassifierRetriever(
            vectorizer=word2vec_vectorizer,
            clf=linear_model.LogisticRegression(solver='lbfgs')
        ),
    'classifier_retriever': retriever.ClassifierRetriever(
        vectorizer=vectorizer.BM25Vectorizer(),
        clf=linear_model.LogisticRegression(penalty='l1', solver='liblinear')
    )
}