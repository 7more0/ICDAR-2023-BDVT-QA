import os
import pandas as pd
import openai
from time import sleep
import re
from similarity.normalized_levenshtein import NormalizedLevenshtein
from tqdm import tqdm


csv_path = "data/v3QA_data_gpt.csv"
res_path = 'data/v3QA_data_gpt_davinci.csv'


def qa_text_davinci_003(text, question):
    pattern1 = re.compile(r'\n')
    pattern2 = re.compile(r'Answer:')
    sleep(3)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = """{} From below text:{}""".format(question, text)
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        temperature=1,
        top_p=0.5,
    )
    answer=response['choices'][0]['text']
    if '\n\n' in answer:
        answer = answer.split('\n\n')[1]
    
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
        qadata["predict"][i] = qa_text_davinci_003(question, text)
    qadata.to_csv(to_data_path, index=False)
    return qadata


def calanls(data_path):
    qadata = pd.read_csv(data_path)
    nls = NormalizedLevenshtein()

    total_nls = 0
    for i in tqdm(range(len(qadata))):
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
        
    return total_nls / len(qadata)


if __name__ == '__main__':
    # get_predict(csv_path, res_path)
    print(calanls(res_path))