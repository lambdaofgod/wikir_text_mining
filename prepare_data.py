import pandas as pd
from nltk import stem
import nltk


stop_words = set(nltk.corpus.stopwords.words('english'))


def stem_text(doc, stemmer=stem.PorterStemmer()):
    stems = [stemmer.stem(elem) for elem in doc.split(" ") if elem not in stop_words]
    return ' '.join(stems)


stemmer = nltk.stem.PorterStemmer()
documents_df = pd.read_csv('wikIR1k/documents.csv', index_col='id_right')
documents_df['text'] = documents_df['text_right']
documents_df['stemmed_text'] = documents_df['text'].apply(lambda doc: ' '.join([stemmer.stem(w) for w in doc.split()]))
documents_df.to_csv('wikIR1k/documents_preprocessed.csv')