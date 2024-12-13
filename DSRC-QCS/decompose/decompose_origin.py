from pick_example import similarity
import re
import re
import torch

global client


def init_client(p_client):
    global client
    client = p_client


def decompose_question(
    device,
    question,
    data_name,
    decompose_examples,
    exam_data,
    model=None,
    tokenizer=None,
    num=1,
    model_name="llama",
    mode="bm25",
    exam_num=2,
    token_cost=0,
):  
    if num > 3:
        return 0, None
    
    prefix = "A complex multi-hop question can be solved step by step by breaking down it into simpler questions. The following are some question decompose examples, where #N represents the answer to question N. If the question is straightforward without needing decomposition, repeat the question itself.\n"
    if exam_num == 0:
            prompt = "Please decompose the multi-hop question into one-hop questions while using #N to represent the answer to one hop question N. If the question is straightforward without needing decomposition, repeat the question itself.\n"
            prompt += "Question: %s\nProblem decomposition:\n" % question
    else:
        prompt = prefix
        prompt += similarity.choose_example(
            question, decompose_examples, exam_data, mode, exam_num
        )

        prompt += "Example %d:\nQuestion: %s\nProblem decomposition:\n" % (
            exam_num + 1,
            question,
        )
    res = generate(
        device,
        model,
        tokenizer,
        prompt,
        max_tokens=100,
        model_name=model_name,
        token_cost=token_cost,
    )
    try:
        tmp_res = re.split("[Ee]xample", res)[0]
        if re.findall("\d\.([^\n]*)", tmp_res):
            res = tmp_res.strip()
        res += "\n"
        questions = re.findall("\d\.([^\n]*)", res)
        tmp = []
        cnt = 0
        for q in questions:
            if not q or q.isspace():
                continue
            if re.search("#", q) is None:
                tmp.append(q.strip())
            cnt += 1
        if not tmp or not len(tmp) :
            return decompose_question(
                device=device,
                model=model,
                question=question,
                data_name=data_name,
                decompose_examples=decompose_examples,
                exam_data=exam_data,
                tokenizer=tokenizer,
                num=num + 1,
                model_name=model_name,
                mode=mode,
                exam_num=exam_num,
                token_cost=token_cost,
            ) 
        q1 = question.lower().strip()
        for q in tmp:
            q = q.lower()
            # 使用正则表达式匹配并删除特殊字符、标点符号和空格
            pattern = re.compile("[^a-zA-Z]")
            q = pattern.sub("", q)
            q1 = pattern.sub("", q1)
            if q1 in q:
                return 0, None
        print("Decompose:\n")
        print(tmp)
        return cnt, tmp
    except Exception as e:
        print(e)
        return decompose_question(
                device=device,
                model=model,
                question=question,
                data_name=data_name,
                decompose_examples=decompose_examples,
                exam_data=exam_data,
                tokenizer=tokenizer,
                num=num + 1,
                model_name=model_name,
                mode=mode,
                exam_num=exam_num,
                token_cost=token_cost,
            )


def generate(
    device,
    model,
    tokenizer,
    prompt,
    max_tokens=100,
    num=1,
    model_name="alpaca",
    token_cost=0,
):
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

        if token_cost > 80:
            print("Token cost is too high")
            return None
        try:
            global client
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
            return generate(
                device,
                model,
                tokenizer,
                prompt,
                max_tokens,
                num + 1,
                model_name=model_name,
                token_cost=token_cost,
            )
