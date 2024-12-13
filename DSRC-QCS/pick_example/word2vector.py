import numpy as np
from gensim.models import KeyedVectors
from gensim.downloader import load
import numpy as np

# 加载预训练的 Word2Vec 模型
model_name = "/model/word2vect-google"
model_path = (
    "/model/word2vect-google/word2vec-google-news-300.model"
)
model = KeyedVectors.load(model_path)

# 加载预计算的向量矩阵
vectors_path = "/model/word2vect-google/word2vec-google-news-300.model.vectors.npy"
vectors = np.load(vectors_path)

# 将向量矩阵设置为模型的向量
model.vectors = vectors


# 将句子中的每个词转换为向量，并计算平均向量
def sentence_vector(sentence):
    words = [word for word in sentence if word in model]  # 移除不在模型中的词
    if not words:
        return np.zeros(model.vector_size)  # 如果句子中没有词在模型中，则返回零向量
    return np.mean(model[words], axis=0)


# 计算两个句子向量的余弦相似度
def similarity_score(query, sentences):
    vec1 = sentence_vector(query)
    vecList = []
    for sentence in sentences:
        vecList.append(sentence_vector(sentence))

    return cosine_similarity(vec1, vecList)


def cosine_similarity(vector, vector_list):
    # 将输入转换为NumPy数组
    vector = np.array(vector)
    vector_list = np.array(vector_list)

    # 计算点积
    dot_products = np.dot(vector_list, vector)

    # 计算范数
    vector_norm = np.linalg.norm(vector)
    vector_list_norms = np.linalg.norm(vector_list, axis=1)

    # 计算余弦相似度
    cosine_similarities = dot_products / (vector_norm * vector_list_norms)

    return cosine_similarities
