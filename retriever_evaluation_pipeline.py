import pandas as pd
import tqdm

from nltk import stem
from nltk.corpus import stopwords
import config
from wikir_text_mining import evaluation


# parameters used by guild.ai (experiment runner)
retriever_type = 'topic_model_retriever'
subset = 'training'
alpha = 0.5
text_col = 'text'
vectorizer = 'bm25'
word_embedding_model = 'glove-wiki-gigaword-50'
word_embedding_classifier = 'linear'
stem_query = True
query_expander_n_expanded_words = 10


# setup
retriever = config.predefined_retrievers(
    retriever_type,
    text_col=text_col,
    alpha=alpha,
    vectorizer=vectorizer,
    word_embedding_model=word_embedding_model,
    query_expander_n_expanded_words=query_expander_n_expanded_words,
    word_embedding_classifier=word_embedding_classifier
)
queries_df = pd.read_csv('wikIR1k/{}/queries.csv'.format(subset), index_col='id_left')
qrel_path = 'wikIR1k/{}/qrels'.format(subset)
evaluator = evaluation.setup_evaluator_from_relevance_file(qrel_path)


stop_words = set(stopwords.words('english'))


def make_query(query_text, stem_query=stem_query):
    if stem_query:
        return [stem.PorterStemmer().stem(w) for w in query_text.split() if w not in stop_words]
    else:
        return [w for w in query_text.split() if w not in stop_words]


retriever_scores = [
    evaluation.evaluate_query(query_id, make_query(query_text), retriever, evaluator)
    for query_id, query_text in tqdm.tqdm(queries_df.itertuples(), total=queries_df.shape[0])
]

print(pd.concat(retriever_scores).mean())
