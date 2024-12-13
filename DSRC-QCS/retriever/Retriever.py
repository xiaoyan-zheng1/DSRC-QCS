import numpy as np
import string
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from retriever import Split

from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer


class Retriever:
    retri_tokenizer = None
    retriever = None
    device = None
    ctx_tokenizer = None
    question_model = None
    ctx_model = None
    question_tokenizer = None

    def mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def __init__(self, mode, device):
        Retriever.device = device
        if mode == "contriever":
            Retriever.retri_tokenizer = AutoTokenizer.from_pretrained(
                "/baseline/baseline_visconde/contriever"
            )
            Retriever.retriever = AutoModel.from_pretrained(
                "/baseline/baseline_visconde/contriever"
            )
            Retriever.retriever.to(device)
        elif mode == "colbert":
            Retriever.retri_tokenizer = AutoTokenizer.from_pretrained(
                "/model/colbert"
            )
            Retriever.retriever = AutoModel.from_pretrained(
                "/model/colbert"
            )
            Retriever.retriever.to(device)
        elif mode == "dpr":
            # 指定模型的本地路径

            ctx_model_path = (
                "/model/dpr-ctx_encoder-single-nq-base"
            )
            question_model_path = (
                "/model/dpr-question_encoder-single-nq-base"
            )

            # 加载 context encoder 和 tokenizer
            Retriever.retriever = DPRContextEncoder.from_pretrained(ctx_model_path).to(
                device
            )
            Retriever.retri_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
                ctx_model_path
            )

            # 加载 question encoder 和 tokenizer
            Retriever.question_model = DPRQuestionEncoder.from_pretrained(
                question_model_path
            ).to(device)
            Retriever.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
                question_model_path
            )

    def normalize(s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        if not s:
            return ""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):

            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def split_sq(sentence, chunk_size=200, filename="Unknown"):
        return Split.split_long_sentence(sentence, chunk_size, filename)

    def get_embedding(paras, mode):
        paras = [Retriever.normalize(para) for para in paras]
        inputs = Retriever.retri_tokenizer(
            paras, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = inputs.to(Retriever.device)
        with torch.no_grad():
            outputs = Retriever.retriever(**inputs)

        if mode == "contriever":
            embeddings = Retriever.mean_pooling(outputs[0], inputs["attention_mask"])
            embeddings = embeddings.cpu()
        elif mode == "dpr" or mode == "colbert":
            embeddings = outputs.pooler_output
            embeddings = embeddings.cpu()


        return embeddings

    def get_qs_embedding(queries, mode):
        queries = [Retriever.normalize(query) for query in queries]
        if mode == "contriever":
            inputs = Retriever.retri_tokenizer(
                queries, padding=True, truncation=True, return_tensors="pt"
            )
            inputs = inputs.to(Retriever.device)
            with torch.no_grad():
                outputs = Retriever.retriever(**inputs)
            embeddings = Retriever.mean_pooling(outputs[0], inputs["attention_mask"])
            embeddings = embeddings.cpu()
        elif mode == "colbert":
            inputs = Retriever.retri_tokenizer(
                queries, padding=True, truncation=True, return_tensors="pt"
            )
            inputs = inputs.to(Retriever.device)
            with torch.no_grad():
                outputs = Retriever.retriever(**inputs)
            embeddings = outputs.pooler_output
            embeddings = embeddings.cpu()
        elif mode == "dpr":
            inputs = Retriever.question_tokenizer(
                queries, padding=True, truncation=True, return_tensors="pt"
            )
            inputs = inputs.to(Retriever.device)
            with torch.no_grad():
                outputs = Retriever.question_model(**inputs)
            embeddings = outputs.pooler_output
            embeddings = embeddings.cpu()

        return embeddings
