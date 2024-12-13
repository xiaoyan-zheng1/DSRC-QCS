from rank_bm25 import BM25Okapi
from pick_example.word2vector import similarity_score

import numpy as np


def choose_example(query, decompose_examples, exam_data, mode="bm25", num=2):
    if num == 0:
        return ""
    tokenized_sentences = []
    exam_data_tmp = []
    for example in decompose_examples:
            tokenized_sentences.append(example.split())
            exam_data_tmp.append(exam_data[decompose_examples.index(example)])
    tokenized_query = query.split()
    if mode == "bm25":
        bm25 = BM25Okapi(tokenized_sentences)
        scores = bm25.get_scores(tokenized_query)
    elif mode == "word2vector":
        scores = similarity_score(tokenized_query, tokenized_sentences)
    elif mode == "random":
        scores = np.random.rand(len(tokenized_sentences))
    sentence_score_pairs = list(zip(exam_data_tmp, scores))
    sorted_pairs = sorted(sentence_score_pairs, key=lambda x: x[1], reverse=False)[
        -num:
    ]
    prompt = ""
    for idx, data in enumerate(sorted_pairs):
        prompt += "Example %d:\n" % (
            idx + 1
        ) + "Question: %s\nProblem decomposition:\n" % (data[0]["raw"])
        for sid, sub in enumerate(data[0]["decomposed"]):
            prompt += "%d. %s\n" % (sid + 1, sub)
        prompt += "\n"
    return prompt
