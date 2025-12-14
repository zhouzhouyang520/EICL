import torch
import torch.nn.functional as F
import os
import pickle
from tqdm import tqdm

from data_processor.utils import config
from data_processor.utils.loader import *
from data_processor.utils.tools.time_warpper import wrapper_calc_time
from sent_model import SentModel
from emo_model import EmoModel
import random
import re

from data_processor.utils.tools.retriver import Retriver
import emotion_probability_generator as emo_prob_gen

if config.use_gpu:
    device_id = config.device_id 
    os.environ["CUDA_VISOBLE_DEVICES"] = str(device_id)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" 
    if torch.cuda.is_available():
        torch.cuda.set_device(int(device_id))

def link_sentence(data):
    update_sent = []
    for sentence in data:
        sent = " ".join(sentence)
        #print(f"sent: {sent}")
        update_sent.append(sent)
    linked_sent = "</s>".join(update_sent)
    #data = re.sub(r'[\ud800-\udfff]', '', data)
    #linked_sent = data.replace( '\n', '</s>' )
    return linked_sent 

def generate_embed(item_info, model, test):
    if test:
        item_info["context"] = item_info["context"][:200]
    
    max_length = len(item_info["context"])
    result_emb = []
    current_batch = 0
    batch_data = []
    for data in tqdm(item_info["context"]):
        linked_sent = link_sentence(data)
        current_batch += 1
        batch_data.append(linked_sent)
        current_length = len(result_emb)
        if current_batch >= config.batch_size or current_length + current_batch >= max_length:
            current_batch = 0
            #print(f"batch_data: {len(batch_data)}, {batch_data}")
            sent_emb = model.gen_emb(batch_data)
            #print(f"current_length {current_length} sent_emb: {sent_emb}")
            #print("------------------------------")
            result_emb.extend(sent_emb) 
            batch_data = []
    return result_emb

def load_sent_embed(test):
    embed_path = f"{config.data_dir}/{config.embed_name}"

    if os.path.exists(embed_path):
        print(f"Loading embedding from path: {embed_path}")
        with open(embed_path, "rb") as f:
            [emb_tra, emb_val, emb_tst] = pickle.load(f)
    else:
        pairs_tra, pairs_val, pairs_tst, vocab = load_dataset()
        if config.emb_model == "semantic":
            model = SentModel(config.use_gpu, config.device_id)
        else:
            # Emotion
            model = EmoModel()
        emb_tra = generate_embed(pairs_tra, model, test)
        emb_val = generate_embed(pairs_val, model, test)
        emb_tst = generate_embed(pairs_tst, model, test)

        with open(embed_path, "wb") as f:
            pickle.dump([emb_tra, emb_val, emb_tst], f)
            print(f"Saving pickle file to path: {embed_path}")

    print(f"emb_tra: {len(emb_tra)}, emb_val: {len(emb_val)}, emb_tst: {len(emb_tst)}")
    return emb_tra, emb_val, emb_tst

def preprocess_ctx(data):
    for i in range(len(data["context"])):
        linked_sent = link_sentence(data["context"][i])
        #print(f"linked_sent: {linked_sent}")
        data["context"][i] = linked_sent 

def construct_emotions(pred_emotions, true_emotion, acc_num):
    emotion_candidates = []
    valid_probs = []
    alpha = 0.2
    sum_prob = 0.0
    print(f"==================================")
    for emo_prob in pred_emotions[:acc_num]: 
        emotion, prob = emo_prob.replace("(", "").replace(")", "").split(",")
        #if sum_prob > threshold: 
        #    break
        emotion_candidates.append(emotion) 
        #print(f"emo_prob: {emo_prob}, float(prob): {float(prob)}, alpha: {alpha}")
        float_prob = round(float(prob) * alpha, 2)
        sum_prob += float_prob
        valid_probs.append(float_prob)

    if true_emotion is None:
        emo_string = ",".join(emotion_candidates)
    else:
        label_prob = round(1.0 - round(sum_prob, 2), 2)
        if true_emotion in emotion_candidates:
            label_index = emotion_candidates.index(true_emotion)
            #print(f"true_emotion: {true_emotion}, emotion_candidates: {emotion_candidates}, label_index: {label_index}")
            #print(f"label_prob: {label_prob}")
            #print(f"valid_probs: {valid_probs}")
            valid_probs[label_index] = round(valid_probs[label_index] + label_prob, 2)
            #print(f"valid_probs after: {valid_probs}")
        else:
            #print(f"before true_emotion: {true_emotion}, emotion_candidates: {emotion_candidates}, valid_probs: {valid_probs}")
            emotion_candidates.insert(0, true_emotion)
            emotion_candidates = emotion_candidates[:acc_num]
            valid_probs.insert(0, label_prob)
            #print(f"after true_emotion: {true_emotion}, emotion_candidates: {emotion_candidates}, valid_probs: {valid_probs}")
        zipped = list(zip(emotion_candidates, valid_probs))
        emo_string = ",".join(list(map(lambda x: "(" + x[0] + "," + str(x[1]) + ")", zipped)))
    print(f"==================================")
        
    return emo_string 
    

def format_data(data, i, example=False, emo_data=None):
    emotion = data["emotion"][i].lower()
    context = data["context"][i]
    target = data["target"]

    pred_emotions = emo_data[i]
#    threshold = 0.9
#    sum_prob = 0.0
#    emotion_candidates = []
#    for emo_prob in pred_emotions[:2]: 
#        emotion, prob = emo_prob.replace("(", "").replace(")", "").split(",")
#        if sum_prob > threshold: 
#            break
#        sum_prob += float(prob)
#        emotion_candidates.append(emotion) 
#    emo_string = ",".join(emotion_candidates)

#    print("[situation]:", " ".join(data["situation"][i]))
#    print("[emotion]:", emotion)
#    print("[context]:", context)
#    print("[target]:", " ".join(target))
    output = ""
    if example:
        prefix = "Dialogue example"
    else:
        prefix = "Dialogue context"
        emo_string = construct_emotions(pred_emotions, None, 2)
        output += f"{emo_string}\n"
    output += f"{prefix}: {context}"

    if example:
        #print("[context]:", context)
        #print(f"train_data", train_data["input"])
        #print("----------------------")
        #output += f"\tTrue emotion label: {emotion}"

#        acc_num = 2
#        pred_emotions = emo_data[i][:acc_num]
#        emo_string = ",".join(pred_emotions)
    
        emo_string = construct_emotions(pred_emotions, emotion, 5)
        output += f"\tPredicted emotion and probability: {emo_string}"
    return output

def load_json(path):
    json_data = json.load(open(path, "r", encoding="utf-8"))
    return json_data

@wrapper_calc_time()
def build_tst_data(indices, test, example_num, emo_tra, emo_tst):
    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset()

    preprocess_ctx(pairs_tra)
    preprocess_ctx(pairs_tst)
    train_ctx = pairs_tra
#    if test:
#        train_ctx = pairs_tst
    length = len(pairs_tra["context"])
    print(length)
    #print(f"Train data length: {length}")
    all_index = list(range(0, length))

    json_list = []
    for i in range(len(indices)):
        json_dict = {"instruction": ""}
        update_sent = []

        (index, weight) = indices[i]
        index, weight = index[0], weight[0]
        #print(f"index: {len(index)}")
        #print(f"weight: {len(weight)}")
        index = index[:example_num]
        weight = weight[:example_num]
        #index = random.sample(all_index, example_num) 

        ctx = format_data(pairs_tst, i, False, emo_tst)
        if test:
            print("==============================")
            print("==============================")
            print(f"i: {i} Context: {ctx}")
            print("==============================")
            print("==============================")
        examples = []
        for k, j in enumerate(index):
            example = format_data(train_ctx, j, True, emo_tra)
            if test:
                print("------------------------------")
                print(f"j:{j} w: {weight[k]}, Example: {example}")
                print("------------------------------")
            examples.append(example)
        if test:
            print(f"examples: {len(examples)}")
            print("==============================")
        json_dict["examples"] = "\n".join(examples)
        json_dict["input"] = ctx
    
        # Target
        target_list = pairs_tst["target"][i]
        target_text = " ".join(target_list)
        emotion = pairs_tst["emotion"][i]
        # json_dict["output"] = f"Emotion:{emotion}</s> Response:{target_text}"
        json_dict[ "output" ] = f"Emotion:{emotion}"
        json_dict["emotion"] = emotion 
        json_list.append(json_dict) 

    print(f"Save Empathetic Dialogue test data...")
    save_path = f"{config.ed_dir}/{config.ed_tst}"
    save_data(json_list, save_path)

def build_exampled_index(test, example_num):
    print(f"Loading sentence embedding.")
    emb_tra, emb_val, emb_tst = load_sent_embed(test)
    retriver = Retriver(topk=50, train_vectors=emb_tra, test_vectors=emb_tst)
    print(f"Searching similar example, and the example number is: {example_num}")
    example_index = retriver.search(emb_tst)
    return example_index 
        
def load_example_index(test, example_num):
    example_path = f"{config.data_dir}/{config.example_name}"

    if os.path.exists(example_path):
        print(f"Loading example index from path: {example_path}")
        with open(example_path, "rb") as f:
            example_index = pickle.load(f)
        old_example_num = len(example_index[0][0][0])
        if old_example_num < example_num:
            print(f"The old_example_num {old_example_num} is not equal to the example_num {example_num}")
            example_index = build_exampled_index(test, example_num)
    else:
        print(f"Example index is not exists in path: {example_path}")
        example_index = build_exampled_index(test, example_num)
        with open(example_path, "wb") as f:
            pickle.dump(example_index, f)
            print(f"Saving example index file to path: {example_path}")
    return example_index 

@wrapper_calc_time()
def convert_data():
    test = True
    test = False
    example_num = config.example_num
    example_index = load_example_index(test, example_num)
    print(f"example_index: {len(example_index)}")

    emo_prob_path = f"{config.ed_dir}/{config.emo_prob_name}"
    print("Loading Emotion with probability data from path: {emo_prob_path}")
    if not os.path.exists(emo_prob_path):
        print(f"File is not exists: {emo_prob_path}")
        emo_prob_gen.generate_probabilities(None, None, None)
    [emo_tra, emo_val, emo_tst] = load_json(emo_prob_path)

    print("Converting index to context text...")
    build_tst_data(example_index, True, example_num, emo_tra, emo_tst)
    print("Converting index to context text end...")

def merge_example_prob():
    basd_dir = "ed_json_data/"
    print("Loading json data...")
    test_prob_data = load_json(basd_dir + "ed_tst_acc_prob.json")
    test_data = load_json(basd_dir + "ed_tst.json")
    print("Loading json data end")
    for i, data in enumerate(test_prob_data): 
        test_data[i]["input"] = data["input"]

    print(f"Save Empathetic Dialogue test data...")
    save_path = f"{config.ed_dir}/{config.ed_tst}"
    save_data(test_data, save_path)

if __name__ == "__main__":
    convert_data()
    #merge_example_prob()
