'''
Author: zhanwei xu
Date: 2023-07-31 21:53:56
LastEditors: zhanwei xu
LastEditTime: 2023-12-28 10:42:02
Description: 

Copyright (c) 2023 by zhanwei xu, Tsinghua University, All Rights Reserved. 
'''
import requests
import json
import pickle
import os
import numpy as np
import datetime
import time
from tqdm import tqdm
from cal_metrics import cal_metrics
import jsonlines

UNK_idx = 0 
PAD_idx = 1 
EOS_idx = 2 
SOS_idx = 3 
USR_idx = 4 
SYS_idx = 5 
CLS_idx = 6 

# conda activate yz_seek

def convert(n):
    return str(datetime.timedelta(seconds = n)) 

def wrapper_calc_time(print_log=True):
    """ 
    :param print_log: 
    :return:
    """

    def wrapper(func):
        def inner_wrapper(*args, **kwargs):
            start_time = time.time()
            func_re = func(*args, **kwargs)
            run_time = time.time() - start_time
            #re_time = f'{func.__name__} execution time: {int(tem_time * 1000)}ms'
            converted_time = convert(run_time)
            if print_log:
                print(f"{func.__name__} time:", run_time, converted_time)
            return func_re

        return inner_wrapper

    return wrapper

class ChatApi():
    def __init__(self, ):
        self.url = "https://api.gptapi.us/v1/chat/completions" #"https://www.gaosijiaoyu.cn/message_key_json"
        OPENAI_API_KEY = "sk-5SuPvQyu0mbZ6e8B1fEb8aD38c9740B196F5DeC756Ab8746"

        self.message_json = {
#           "message": [
#              {
#                 "role": "user",
#                 "content": "Hello, are you gpt3.5 or gpt4?"
#              }
#           ],
#           "model": "gpt-3.5-turbo",

            #"model": "claude-3-haiku",
           "model": "gpt-3.5-turbo",
            "messages": [
              {
                "role": "user",
                "content": "You are a helpful assistant."
              }
            ]
        }
        self.headers = {
           'Content-Type': 'application/json',
            'Authorization': f"Bearer {OPENAI_API_KEY}" 
        }
        print(f"self.headers: {self.headers}")

    def request_data(self, data):
        self.message_json["messages"][0]["content"] = data
        payload = json.dumps(self.message_json)
        #print(f"payload: {payload}")
        response = requests.request("POST", self.url, headers=self.headers, data=payload, stream=True)
        #print(response.status_code)
        if response.status_code == 200:
            result = json.loads(response.text)
            #print(f"response.text: {response.text}")
            res = result['choices'][0]['message']['content']
            #print("---------------------")
            #print("request data:", res)
            #print("---------------------")
#        if response.status_code == 200:
#            res = []
#            for chunk in response.iter_content(chunk_size=512):
#                print(chunk)
#                word = chunk.decode('utf-8')
#                res.append(word)
#            print("request data:", "".join(res))
#            res = "".join(res)
#            print("=====---------------------")
#            print("request data:", res["choices"])
#            print("=====---------------------")
        else:
            res = ""
        return res, response.status_code

def read_instruction(data_path="ed_json_data/instruction.txt"):
    with open(data_path, 'r') as f:
        content = f.read()
        #print(f"Instruction: {content}")
        return content 


#def load_dataset():
#    cache_file = "data/ED/dataset_preproc.p"
#    #if os.path.exists(cache_file):
#    print("LOADING empathetic_dialogue")
#    with open(cache_file, "rb") as f:
#        #[data_tra, data_val, data_tst, vocab] = pickle.load(f)
#        data_tra, data_val, data_tst, _ = pickle.load(f)
#
#    for i in range(10):
##    for i in [18257, 32799, 39114, 5065, 39650]:
#        print("[situation]:", " ".join(data_tra["situation"][i]))
#        print("[emotion]:", data_tra["emotion"][i])
#        print("[context]:", [" ".join(u) for u in data_tra["context"][i]])
#        print("[target]:", " ".join(data_tra["target"][i]))
#        print(" ")
#
##    print(f"data test start ========================")
##    for i in range(len(data_tst["context"])):
##        print("[situation]:", " ".join(data_tst["situation"][i]))
##        print("[emotion]:", data_tst["emotion"][i])
##        print("[context]:", [" ".join(u) for u in data_tst["context"][i]])
##        print("[target]:", " ".join(data_tst["target"][i]))
##        print(" ")
##    print(f"data test end ========================")
#
##
##    print("============================")
##    print("Check test data")
##    for i in [333, 543, 124, 100, 5000]:
##        print("[situation]:", " ".join(data_tst["situation"][i]))
##        print("[emotion]:", data_tst["emotion"][i])
##        print("[context]:", [" ".join(u) for u in data_tst["context"][i]])
##        print("[target]:", " ".join(data_tst["target"][i]))
##        print(" ")
#    return data_tra, data_val, data_tst, vocab

def load_json(path):
    json_data = json.load(open(path, "r", encoding="utf-8"))
    return json_data

def save_data(json_data, path):
    with open(path, mode='w') as file:
        writer = jsonlines.Writer(file)
        for item in json_data:
            writer.write(item)

#    data_file = open(path, 'w', encoding='utf-8')
#    json.dump(json_data, data_file, indent=4)

@wrapper_calc_time(print_log=True)
def chat_test_data(chat_api, instruction, test_data, test):
    results = []
    retries = 20
    for data in tqdm(test_data):
        one_data = {}
        #data = "Hello, are you support batch size for multiply dialogue?"
        input_data = data["input"]
        output_data = data["output"]
        #input_data = instruction + input_data
        examples = data["examples"]
        # print(examples)
        #emo_vad = data["emotion_vad"]
        #.format(emo_vad)
        tmp_instruction = "Predicted emotions for dialogue context:"
        input_data = instruction + examples + "\n" + tmp_instruction + input_data 
        if test:
            print(f"input_data: {input_data}")
            print("=============================")
#        print(f"output_data: {output_data}")

        chat_out = None
        status_code = 200
#        chat_out, status_code = chat_api.request_data(input_data)
        for attempt in range(retries):
            try:
                chat_out, status_code = chat_api.request_data(input_data)
            except Exception as e:  # Replace Exception with specific exceptions the client might raise
                if attempt < retries - 1:
                    if attempt > 3:
                        attempt = 3
                    time.sleep(2 ** attempt)  # Exponential back-off
                    continue
                else:
                    chat_out = None
            if status_code == 200 and chat_out is not None:
                break

#        while True:
#            try:
#                chat_out, status_code = chat_api.request_data(input_data)
#            except Exception as e:
#                print(e)
#            if status_code == 200:
#                break;
#        print(f"chat_out: {chat_out}")
#        print("===========================")
        one_data["label"] = "[gMASK]sop " + output_data 
        one_data["predict"] = chat_out
        results.append(one_data)
    return results

def run():
    test_flag = True
    test_flag = False
    basd_dir = "ed_json_data/"
    test_data = load_json(basd_dir + "ed_tst.json")
    if test_flag:
        test_data = test_data[:2]
    instruction = read_instruction()

#    tmp_data = []
#    for data in test_data:
#        if data["output"] in ["Emotion:trusting\n Response:wow ! a 20 foot boat , that must be amazing to take out on the lake ! how much did it cost you ?", "Emotion:trusting\n Response:i am always leery of loaning money to friends", "Emotion:disappointed\n Response:they do . i can not believe how fast the time goes .", "Emotion:ashamed\n Response:maybe they will understand .", "Emotion:confident\n Response:what subject is it ?", "Emotion:faithful\n Response:what is your most successful campaign so far ?"]:
#            tmp_data.append(data)
#    print(f"tmp_data: {len(tmp_data)}")
#    test_data = tmp_data

    #print(f"instruction: {instruction}")
    print(f"Test data length: {len(test_data)}")
    chat_api = ChatApi()
    print(f"Chat test data...")
    results = chat_test_data(chat_api, instruction, test_data, test_flag)
    print(f"Save data...")
    save_data(results, "output_ed/generated_predictions.jsonl")
    print(f"Cal metrics...")
    cal_metrics()

#cal_metrics()
run()

