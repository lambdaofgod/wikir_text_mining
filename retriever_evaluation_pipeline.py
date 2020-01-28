import pandas as pd
import tqdm

import config
from wikir_text_mining import evaluation

retriever_type = 'word2vec_classifier_retriever'
subset = 'training'
retriever = config.PREDEFINED_RETRIEVERS[retriever_type]


queries_df = pd.read_csv('wikIR1k/{}/queries.csv'.format(subset), index_col='id_left')

qrel_path = 'wikIR1k/{}/qrels'.format(subset)
evaluator = evaluation.setup_evaluator_from_relevance_file(qrel_path)


retriever_scores = [
    evaluation.evaluate_query(query_id, query_text.split(), retriever, evaluator)
    for query_id, query_text in tqdm.tqdm(queries_df.itertuples(), total=queries_df.shape[0])
]

print(pd.concat(retriever_scores).mean())