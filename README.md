# wikir_text_mining

Using text mining and machine learning for improving standard Information Retrieval

## Motivation
One of the first things that data scientists do when facing a new problem is searching for related problems and software on GitHub. GitHub's search capabilities are very limited (at the time of writing this report) - there are several features like project descriptions, tags and readmes that could be helpful, but its search engine doesn't allow for searching them simultaneously. The effect is that sometimes reformulating query might result in drastically different search results. This project aims at evaluating in a controlled experiment several simple text mining methods that can be useful for expanding traditional bag-of-words retrieval model.

## Abstract
WikIR is a recently proposed dataset for evaluating Information Retrieval. So far it was used only to benchmark standard Information Retrieval method (BM25)\citep{manning2008introduction} versus deep learning-based text matching. In this project several machine learning methods for text, using classification, word embeddings and matrix decomposition were used to improve searching results.

## Running

The project uses [guild.ai](https://github.com/guildai/guildai) for running and managing experiments.

Run an experiment

`
guild run retriever_evaluation_pipeline {ARGS}
`

See `retriever_evaluation_pipeline` for all parameters.

Most important parameters:

*retriever_type*: one of 

    * baseline
    * *ord_embedding_classifier_retriever
    * classifier_retriever
    * naive_bayes_classifier_retriever 
    * topic_model_retriever
    * query_expander

*subset*: *training*, *validation*, *test*

*text_col*: text column used for reranking/classification. One of *text*, *stemmed_text*, *keywords*