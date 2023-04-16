import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import DistilBertTokenizer, DistilBertModel
import re
from similarity.normalized_levenshtein import NormalizedLevenshtein
from tqdm import tqdm


csv_path = "data/v3QA_data_gpt.csv"
res_path = 'data/v3QA_data_gpt_bert.csv'


def qa_bert(text, question):
    """
    param:
    question: 问题文本,str
    text: 内容文本,str
    return: 预测的回答文本,str
    """
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    
    # 将问题和文本token化
    input_ids = tokenizer.encode(question, text)
    
    #字符串版本
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    #段IDs
    #first occurence of [SEP] token
    sep_idx = input_ids.index(tokenizer.sep_token_id)
    
    # 段A中的token数
    num_seg_a = sep_idx+1
    
    # 段B中的token数
    num_seg_b = len(input_ids) - num_seg_a
    
    # 段嵌入的0和1列表
    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    assert len(segment_ids) == len(input_ids)
    
    # 使用input_ids和segment_ids的模型输出
    try:
        output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))
    except RuntimeError:
        return "Unable to find the answer to your question."
    # 重建答案
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)
    
    answer = "Unable to find the answer to your question."
    if answer_end >= answer_start:
        answer = tokens[answer_start]
        for i in range(answer_start+1, answer_end+1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]
                
    if answer.startswith("[CLS]"):
        answer = "Unable to find the answer to your question."
    
    pattern1 = re.compile(r'\n')
    pattern2 = re.compile(r'Answer:')
    ans = pattern1.sub('', str(answer))
    ans = pattern2.sub('', ans).strip()
    
    if ans.startswith('Yes'):
        ans = 'Yes.'
    elif ans.startswith('No'):
        ans = 'No.'
    else:
        pass
    return ans


def get_predict(data_path,to_data_path):
    qadata = pd.read_csv(data_path)
    # qadata.head()
    qadata["predict"] = 'Unsure about answer.'

    for i in tqdm(range(len(qadata))):
        question = qadata["question"][i]
        text = qadata["text"][i]
        qadata["predict"][i] = qa_bert(question, text)
    qadata.to_csv(to_data_path, index=False)
    return qadata


def calanls(data_path):
    qadata = pd.read_csv(data_path)
    nls = NormalizedLevenshtein()

    total_nls = 0
    for i in range(len(qadata)):
        temp = 0
        if isinstance(qadata['answer'][i], str):
            answers = eval(qadata['answer'][i])
        elif isinstance(qadata['answer'][i], list):
            pass
        else:
            KeyError('Check your answer type.')
        for j in range(len(answers)):
            temp = max(nls.similarity(answers[j], str(qadata['predict'][i])),temp)
        total_nls += temp
        # print(temp)
    
    return total_nls / len(qadata)


if __name__ == '__main__':
    get_predict(csv_path, res_path)
    print(calanls(res_path))