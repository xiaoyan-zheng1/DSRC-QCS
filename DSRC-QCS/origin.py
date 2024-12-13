import argparse
from openai import OpenAI
import json
import string
import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from datasets import load_dataset
import re
import torch
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize
from rank_bm25 import BM25Okapi
import math
from collections import Counter
from retriever.Retriever import Retriever
from decompose.decompose_origin import decompose_question as decompose_question
from decompose.decompose_origin import init_client as init_decompose_client

num_decompose = 2
results = {}
global token_cost
token_cost = 0
regex = r"([。？！；\n.!?;]\s*)"


# 定义模型参数
class ModelArguments:
    def __init__(self, model_name_or_path):
        self.model_name_or_path = model_name_or_path


def generate(prompt, max_tokens=100, num=1, model_name="alpaca"):
    if model_name != "gpt":
        # 使用模型生成回复
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(device)
        with torch.no_grad():
            generate_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                min_new_tokens=1,
                num_return_sequences=1,
            )
        print("Answer generated. Decoding........")
        response = tokenizer.decode(generate_ids[0])
        response = response.replace(prompt, "")
        response = response.replace("<s>", "")
        response = response.replace("</s>", "")

        return response
    else:
        if num > 3:
            return None
        global token_cost
        if token_cost > 80:
            print("Token cost is too high")
            return None
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0613",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=max_tokens,
            )
            token_cost += response.usage.total_tokens / 1000 * 0.002
            return response.choices[0].message.content
        except Exception as e:
            print(e)
            return generate(prompt, max_tokens, num + 1, model_name=model_name)


def get_response(question, segments, model_name="llama", positive=True):
    prompt = ""
    context = ""

    if len(segments) > 1:
        context = "\n".join(segments)
        if positive:
            prompt = (
                "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n"
                + context
                + "\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: "
                + question
                + "\nAnswer:"
            )

        else:
            subffix = "Can the question be solved according to the passages? Please firstly reason step by step, then output the answer. It's possible that the passages don't contain relevant information to address the question."
            prompt = context + "Question:%s\n%s\nResponse:" % (question, subffix)
    else:
        context = segments[0]
        if positive:
            prompt = (
                "Answer the question based on the given passage. Only give me the answer and do not output any other words.\n\nThe following are given passage.\n"
                + context
                + "\n\nAnswer the question based on the given passage. Only give me the answer and do not output any other words.\n\nQuestion: "
                + question
                + "\nAnswer:"
            )

        else:
            subffix = "Can the question be solved according to the passage? Please firstly reason step by step, then output the answer. It's possible that the passages don't contain relevant information to address the question."
            prompt = context + "Question:%s\n%s\nResponse:" % (question, subffix)
  
    res = generate(prompt, model_name=model_name)
    if not res:
        return None, None
    if re.match(r"[.\s\S]*Please", res):
        res = re.match(r"([.\s\S]*)Please", res).group(1)
    if "none" in res.lower():
        return res.strip(), None
    if positive and (model_name == "gpt" or len(res.split()) < 10):
        return res.strip(), res.strip()
    subffix = "Remove the redundant words in response and output a concise answer directly with 'Answer: '. If the answer does not exist, the output should be 'Answer: None'."
    prompt = "Question:%s\nResponse:%s\n" % (question, res) + subffix + "\nAnswer:"
    explaination = res
    answer = generate(prompt, model_name=model_name)
    if "none" in answer.lower():
        return explaination, None

    return explaination, answer


def cal_f1(question, prediction, ground_truths):
    if not prediction or prediction.isspace():
        return 0

    def normalize_answer(s):
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

    def f1_score(prediction, ground_truth):
        common = Counter(prediction) & Counter(ground_truth)
        # collections.Counter计算每个元素出现的次数，得到如下结果：Counter({'x': 4, 'y': 2, 'z': 2})
        # Counter('abb')&Counter('bcc') 结果：Counter({'b':1})
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction)
        recall = 1.0 * num_same / len(ground_truth)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def qa_f1_score(question, prediction, ground_truth):
        if not prediction or not ground_truth:
            return 0
        normalized_ground_truth = normalize_answer(ground_truth)
        ground_truth_tokens = normalized_ground_truth.split()
         
        prompt = "Remove the redundant words in response and output a concise answer directly with 'Answer: '.\n"
        prompt += "Question:%s\nResponse:%s\n" % (question, prediction)
        prediction = generate(prompt, model_name=model_name)
        prediction = re.sub(r"\b(answer|Answer)\b", " ", prediction)
        normalized_prediction = normalize_answer(prediction)
        prediction_tokens = normalized_prediction.split()
        tmp_score = f1_score(prediction_tokens, ground_truth_tokens)

        return tmp_score

    score = 0.0
    if isinstance(ground_truths, list):  # 答案可能有别名
        for ground_truth in ground_truths:
            score1 = qa_f1_score(question, prediction, ground_truth)
            if score <= score1:
                score = score1

    else:
        score = qa_f1_score(question, prediction, ground_truths)
    return score




class LeafNode:
    def __init__(self, question, model_name):

        self.question = question
        self.answer = None
        self.answer1 = None
        self.explaination = None
        self.children = []
        self.substitutor = None
        self.gold_answer = None
        self.answers = None
        self.segments = None
        print("get_response of question:%s" % question)
       
        top_titles, final_pairs = retriever_k(question, top_k_segment, args.retriever)
        self.doc_rank = top_titles
        tmp_segments, answers, explainations = [], [], []
        answer, explaination = None, None
        tmp_titles = []
        tmp_segments = []
        for p in final_pairs:
            if p[0].strip() in tmp_titles:
                tmp_segments[tmp_titles.index(p[0])].append(p[1])
            else:
                tmp_titles.append(p[0].strip())
                tmp_segments.append([p[1]])
        segments = []
        for t, paras in zip(tmp_titles, tmp_segments):
            s = t + ". "
            for p in paras:
                s += p
            segments.append(s)
        self.top_k_segments = segments

        explaination, answer = get_response(
            question, segments, model_name, positive=True
        )
        tmp_segments = [p[1] for p in final_pairs]
        answer = answer.replace("\n", "").strip()
        answers.append(answer)
        explainations.append(explaination)
        self.answer = answer
        self.answer1 = answer
        self.explaination = explaination
        self.answers = answers
        self.segments = tmp_segments

    def get_parent(self):
        return self.parent

    def get_answer(self):
        return self.answer

    def add_child(self, child):
        self.children.append(child)

    def children_num(self):
        return len(self.children)


class TreeNode:  # 非叶子节点
    def __init__(self, question):
        self.question = question
        self.children = []
        self.substitutor = None
        self.answer = None
        self.answer1 = None
        self.explaination = None
        self.gold_answer = None

    def add_child(self, child):
        self.children.append(child)

    def children_num(self):
        return len(self.children)

    def get_answer(self):
        return self.answer



def modify_question(
    question, sub_questions, num=1, model_name="llama"
):
    if num > 3:
        return "error", None
    prefix = "Given Question-Answer pairs, if the question Q can be answered based on the information gained from these pairs, provide the final answer. If not, utilize existing information to rephrase the initial question Q into a simpler equivalent question that contains more known information. Here are examples:\n"
    example1 = "Given:\nQuestion: How long was Schaub's shortest field goal yards?\nAnswer: 12-yard.\nFor question Q: How many yards was the difference between Schaub's longest and shortest field goals?\nOutput: The question Q can not be answered base on the given facts. But we can infer that the question Q is equivalent to: How many yards was the difference between Schaub's longest field goal and 12 yards?\n\n"
    example2 = "Given:\nQuestion: When was Titanic  released?\nAnswer: 1998.\nQuestion: When was Love Actually released?\nAnswer: 2003.\nFor question Q: Which film was released earlier, Titanic or Love Actually?\nOutput: The question Q can be answered base on the given facts. The final answer is: Titanic released earlier.\n\n"
    example3 = "Given:\nQuestion: Who is the father of King Kang Of Zhou?\nAnswer: King Cheng Of Zhou.\nFor question Q: Who is the paternal grandmother of King Kang Of Zhou?\nOutput: The question Q can not be answered base on the given facts. But we can infer that the question Q is equivalent to: Who is the mother of King Cheng Of Zhou?\n\n"
    examples = ""
    exams = [example1, example2, example3]
    if model_name == "gpt":
        exams = [example1, example2]
    for eid, example in enumerate(exams):
        examples += "Example%d\n" % (eid + 1) + example
    query = "Example%d\n" % (len(exams) + 1) + "Given:\n"
    for qs, answer in sub_questions:
        answer = answer.strip(".")
        query += "Question: %s\nAnswer: %s.\n" % (qs, answer)
   
    query += "For question Q: %s\nOutput:" % question
    prompt = prefix + examples + query
    res = generate(prompt, max_tokens=50, model_name=model_name)
    if not res:
        return "error", None
    res = res.strip()
    try:
        tmp_res = re.split("[Ee]xample", res)[0]
        if re.findall("\d\.([^\n]*)", tmp_res):
            res = tmp_res.strip()
    except:
        pass
    if "is equivalent to" in res.lower():
        try:
            next_question = re.search("is equivalent to([^\?\n]*)", res).group(1)

            next_question += "?"
        except Exception:
            next_question = re.search("is equivalent to(.*)", res).group(1)
            if len(next_question.split()) > 30:  
                prompt = (
                    query
                    + res
                    + "\nAccording to above, directly output that question Q "
                    + question
                    + " is equivalent to:"
                )
                next_question = generate(prompt, model_name=model_name)
               
        if not next_question:
            return "error", None
        next_question = next_question.strip(":").strip()
        print("next question:%s" % next_question)
        return "next_question", next_question
    elif "can be answered" in res.lower():
        try:
            answer = re.search("final answer is([^\n]*)", res).group(1)
        except Exception:
            prompt = (
                query
                + res
                + "\nAccording to above, directly output that the final answer for the question Q "
                + question
                + " is:"
            )
            answer = generate(prompt, model_name=model_name)
        if not answer:
            return "error", None
        answer = answer.strip(":").strip()
        return "answer", answer
    else:
        return modify_question(
            question, sub_questions, num + 1, model_name=model_name
        )


def solve(
    question,
    num_try=1,
    model_name="llama",
    mode="bm25",
    exam_num=2,
):
    if num_try > num_decompose:
        root = LeafNode(question, model_name=model_name)
        return root
    cnt, questions = decompose_question(
            device=device,
            model=model,
            question=question,
            data_name=data_name,
            decompose_examples=decompose_examples,
            exam_data=exam_data,
            tokenizer=tokenizer,
            num= 1,
            model_name=model_name,
            mode=mode,
            exam_num=exam_num,
            token_cost=token_cost,
        )  

    if not questions or not len(questions):  # 分解失败
        root = solve(
            question,
            num_try + 1,
            model_name,
            mode=mode,
            exam_num=exam_num,
        )

        return root
    root = TreeNode(question)
    sub_questions = []
    flag = False
    
    for qs in questions:
        node = LeafNode(qs, model_name=model_name)  # 默认这些问题是简单问题
        root.add_child(node)
        if node.answer:
            flag = True
            sub_questions.append((qs, node.answer))
    if flag:
        label, new_qs = modify_question(
            question,
            sub_questions=sub_questions,
            model_name=model_name,
        )
        w_num = 0
        while label == "next_question" and w_num < 2:
            w_num += 1
            if cnt - len(sub_questions) - w_num > 1:
                node = solve(
                    new_qs,
                    model_name=model_name,
                    mode=mode,
                    exam_num=exam_num,
                )  
            else:
                node = LeafNode(new_qs, model_name=model_name)
            root.substitutor = node  
            if node.answer:
                sub_questions.append((new_qs, node.answer))
            label, new_qs = modify_question(
                question,
                sub_questions=sub_questions,
                model_name=model_name,
            )
        if label == "next_question":
            node = LeafNode(new_qs, model_name=model_name)
            root.substitutor = node  
            root.answer = node.answer
        else:
            root.answer = new_qs
        if not root.answer:
            root = LeafNode(question, model_name=model_name)
            return root
        return root
    else:
        root = solve(
            question,
            num_try + 1,
            model_name=model_name,
            mode=mode,
            exam_num=exam_num,
        )
        return root


def save_tree_to_json(node, file_path):
    tree_data = _serialize_tree(node)
    with open(file_path, "w") as file:
        json.dump(tree_data, file, indent=2)


def _serialize_tree(node):
    if not node:
        return None
    if isinstance(node, TreeNode):
        serialized_node = {
            "question": node.question,
            "answer": node.answer,
            "answer1": node.answer1,
            "gold_answer": node.gold_answer,
            "explaination": node.explaination,
        }
    else:
        serialized_node = {
            "question": node.question,
            "answer": node.answer,
            "gold_answer": node.gold_answer,
            "explaination": node.explaination,
            "doc_rank": node.doc_rank,
            "top_k_segments": node.top_k_segments,
            "segments_used": node.segments,
            "answers": node.answers,
        }
    if node.children:
        serialized_node["children"] = [
            _serialize_tree(child) for child in node.children
        ]
    if node.substitutor:
        serialized_node["next"] = _serialize_tree(node.substitutor)
    return serialized_node




def get_docs(contexts):
    docs = re.split(r"Passage [\d]+:", contexts)
    titles = []
    tmp = []
    for e in docs:
        e = e.strip()
        if not e or e.strip().isspace():
            continue
        t = re.match(r"(.*)\n", e).group(1)
        e = re.sub(t, "", e, 1).strip()
        tmp.append(e)
        titles.append(t.strip())

    docs = tmp

    return docs, titles


def retriever_k(question, k=7, mode="contriever"):
    para_scores = []
    doc_scores = []
    question_embedding = Retriever.get_qs_embedding([question], mode)[0]
    scores = []
    for s_embedding in loaded_embeddings:
        scores.append(torch.dot(question_embedding, s_embedding).item())
    point = 0
    for did, paras in enumerate(doc_segments):
        ps_len = 0
        para_scores.append([])
        for pid in range(len(paras)):
            if doc_flags[did][pid]:
                para_scores[-1].append(scores[point])
                ps_len += 1
            else:
                para_scores[-1].append(0)
            point += 1
        if ps_len:
            doc_scores.append(sum(para_scores[-1]) / ps_len)
        else:
            doc_scores.append(-100)
    indexs = sorted(
        list(range(len(doc_scores))), key=lambda x: doc_scores[x], reverse=True
    )
    indexs = indexs[:5]

    top_paras = []
    para_titles = []
    top_para_scores = []
    for index in indexs:
        if doc_scores[index] == -100:
            continue
        para_titles.extend(
            [whole_titles[index] for _ in range(len(doc_segments[index]))]
        )
        top_paras.extend(doc_segments[index])
        top_para_scores.extend(para_scores[index])
    final_pairs = [
        [t, para, score]
        for t, para, score in zip(para_titles, top_paras, top_para_scores)
    ]
    final_pairs = sorted(final_pairs, key=lambda x: x[2], reverse=True)
    final_pairs = final_pairs[: min(k, len(final_pairs))]
    top_titles = list(set([t[0] for t in final_pairs]))
    return top_titles, final_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_name", type=str, default="alpaca")
    parser.add_argument("--pick_algorithm", type=str, default="bm25")
    parser.add_argument("--example_num", type=int, default=2)
    parser.add_argument("--data_name", type=str, default="wikimqa")
    parser.add_argument("--retriever", type=str, default="contriever")
    parser.add_argument("--top_k_segment", type=int, default=7)

    args = parser.parse_args()
    top_k_segment = args.top_k_segment
    device = args.device
    data_name = args.data_name
    model_name = args.model_name
    tokenizer = None
    model = None
    prefix_path = "/data/ablation/%s_top%d_%s_%d_%s_%s/" % (
        args.retriever,
        top_k_segment,
        args.pick_algorithm,
        args.example_num,
        data_name,
        model_name,
    )
    if not os.path.exists(prefix_path):
        os.mkdir(prefix_path)
    decompose_examples = []
    with open(
            "/data/public/decompose_examples.json", "r"
    ) as f:
        exam_data = json.load(f)

    for example in exam_data:
        decompose_examples.append(example["raw"])
    if data_name == "wikimqa":
        path = "/data/public/2wikimqa.jsonl"
    if data_name == "hotpotqa":
        path = "/data/public/hotpotqa.jsonl"
    if data_name == "musique":
        path = "/data/public/musique.jsonl"

    Retriever(args.retriever, device)

    if model_name == "llama":
        model_path = "meta-llama/Llama-2-7b-chat-hf"
        model_args = ModelArguments(model_path)
        # 初始化模型和分词器
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir="",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir="",
        )  
        model.to(args.device)
        model.config.temperature = 0.7
    elif model_name == "gpt":

        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="",
        )
        init_decompose_client(client)
    elif model_name == "alpaca":
        model_path = "/alpaca-7b"
        model_args = ModelArguments(
            model_path
        )  
        # 初始化模型和分词器
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir="",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir="",
        )
        model.to(args.device)

    doc_segments = []
    with open(path, "r") as f:
        lines = f.readlines()
    test_set = []
    for l in lines:
        test_set.append(json.loads(l))



    for data_id, example in enumerate(test_set):  
        context_long = example["context"]
        question = example["input"]
        whole_docs, whole_titles = get_docs(context_long)
        para_embeddings = []
        doc_segments = []
        doc_flags = []
        loaded_embeddings = []
        for doc, title in zip(whole_docs, whole_titles):
            paras = Retriever.split_sq(doc, chunk_size=200)
            doc_segments.append(paras)
            doc_flags.append([True for _ in range(len(paras))])
            paras = ["Title: " + title + "Content: " + p for p in paras]
            loaded_embeddings.extend(Retriever.get_embedding(paras, args.retriever))
        del whole_docs
        print(question)

        root = None
        root = solve(
            question,
            level=1,
            type=type,
            model_name=model_name,
            mode=args.pick_algorithm,
            exam_num=args.example_num,
        ) 
        root.gold_answer = example["answers"]

        save_tree_to_json(root, prefix_path + "%d.json" % data_id)
