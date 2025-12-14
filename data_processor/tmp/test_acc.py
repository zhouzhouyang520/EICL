
from tqdm import tqdm
import pickle
from src.utils import config
from src.utils.data.loader import *
from emo_model import EmoModel
import emotion_probability_generator as emo_prob_gen
from sklearn.metrics import top_k_accuracy_score
import os
from sklearn.metrics import classification_report

#labels = ['afraid', 
#            'angry',
#            'annoyed',
#            'anticipating',
#            'anxious',
#            'apprehensive',
#            'ashamed',
#            'caring',
#            'confident',
#            'content',
#            'devastated',
#            'disappointed',
#            'disgusted',
#            'embarrassed',
#            'excited',
#            'faithful',
#            'furious',
#            'grateful',
#            'guilty',
#            'hopeful',
#            'impressed',
#            'jealous',
#            'joyful',
#            'lonely',
#            'nostalgic',
#            'prepared',
#            'proud',
#            'sad',
#            'sentimental',
#            'surprised',
#            'terrified',
#            'trusting']

labels = ['afraid', 
            'angry',
            'annoyed',
            'anticipating',
            'anxious',
            'apprehensive',
            'ashamed',
            'caring',
            'confident',
            'content',
            'devastated',
            'disappointed',
            'disgusted',
            'embarrassed',
            'excited',
            'faithful',
            'furious',
            'grateful',
            'guilty',
            'hopeful',
            'impressed',
            'jealous',
            'joyful',
            'lonely',
            'nostalgic',
            'prepared',
            'proud',
            'sad',
            'sentimental',
            'surprised',
            'terrified',
            'trusting',
            'agreeing',
            'acknowledging',
            'encouraging',
            'consoling',
            'sympathizing',
            'suggesting',
            'questioning',
            'wishing',
            'neutral', 'disagreeing']

print(f"labels length: {len(labels)}")

#def emo2label(emotion):
#    return labels.index(emotion)

def link_sentence(data):
    update_sent = []
    for sentence in data:
        sent = " ".join(sentence)
        #print(f"sent: {sent}")
        update_sent.append(sent)
    linked_sent = f"</s>".join(update_sent)
    return linked_sent 

def generate_embed(item_info, model, test):
    if test:
        item_info["context"] = item_info["context"][:10]
    
    max_length = len(item_info["context"])
    result_acc = []
    current_batch = 0
    batch_data = []
    batch_label = []
    pred_y = []
    true_y = []
    
    for (i, data) in tqdm(enumerate(item_info["context"])):
        linked_sent = link_sentence(data)
        current_batch += 1
        batch_data.append(linked_sent)
        emotion = item_info["emotion"][i]
        #print(f"emotion: {emotion}, labels.index(emotion): {labels.index(emotion)}")
        batch_label.append(labels.index(emotion))
        current_length = len(result_acc)
        if current_batch >= config.batch_size or current_length + current_batch >= max_length:
            current_batch = 0
            #print(f"batch_data: {len(batch_data)}, {batch_data}")
            emo_logits = model.gen_emb(batch_data, softmax=True)[:, :32]
            pred_program = np.argmax(emo_logits, axis=1)
            zipped = list(zip(list(pred_program), batch_label))
            #print(f"pred_program: {pred_program}, batch_label: {batch_label}, {zipped}")
            #print(f"current_length {current_length} emo_logits: {emo_logits.shape}")
            #print("------------------------------")
            np_emo_logits = emo_logits
            y_true = np.array(batch_label)
            true_y.extend(y_true)
            pred_y.extend(pred_program)
            used_labels = list(range(32))
            program_acc = top_k_accuracy_score(y_true, np_emo_logits, labels=used_labels, k=max_k)
            result_acc.append(program_acc) 
            batch_data = []
            batch_label = []
    return result_acc, true_y, pred_y

def predict(test):
    print("Loading data...")
    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset()
    model = EmoModel()
#    emb_tra = generate_embed(pairs_tra, model, test)
#    emb_val = generate_embed(pairs_val, model, test)
    print("Compute accuracy...")
    acc_list, true_y, pred_y = generate_embed(pairs_val, model, test)
    #acc_list, true_y, pred_y = generate_embed(pairs_tst, model, test)
    #acc_list = generate_embed(pairs_tra, model, test)
    acc = np.mean(np.array(acc_list))
    print(f"acc_list: {acc_list}, acc: {acc}")
    report = classification_report(true_y, pred_y)
    print(report)

    return acc

#test = True
#test = False#True
#result = []
#check_list = [1, 2, 5, 10, 15, 20, 25]
#check_list = [1]
#for i in check_list: 
#    max_k = i 
#    acc = predict(test)
##    result.append((max_k, acc))
##print(f"result: {result}")
    
def load_json(path):
    json_data = json.load(open(path, "r", encoding="utf-8"))
    return json_data

#def test(k, emo_tst, pairs_tst):
#    pred_y = []
#    for pred_emotions in emo_tst:
#        #print(f"pred_emotions: {pred_emotions}")
#        pred_emotions = list(map(lambda x: labels.index(x.replace("(", "").split(",")[0]), pred_emotions))[:k]
#        #print(f"pred_emotions after: {pred_emotions}")
#        #print("-------------------------")
#        pred_y.append(pred_emotions)
#    true_emo_tst = pairs_tst["emotion"]
#    #print(f"true_emo_tst: {true_emo_tst[:10]}")
#    true_y = list(map(lambda x: labels.index(x), true_emo_tst))
#
##    report = classification_report(true_y, pred_y)
##    print(report)
#    #print(f"true_y: {true_y[:10]}")
#    #print("===============")

def test(k, emo_tst, pairs_tst):
    result = []
    true_emo_tst = pairs_tst["emotion"]
    y_true = list(map(lambda x: labels.index(x.lower()), true_emo_tst))
    y_pred = []
    count = 1.0
    for i, pred_emotions in enumerate(emo_tst):
        #print(f"pred_emotions: {pred_emotions}")
        pred_emotions = list(map(lambda x: labels.index(x.replace("(", "").split(",")[0]), pred_emotions))[:k]
        true_emo = y_true[i]
        if true_emo in pred_emotions:
            count += 1.0 
    #print(f"k: {k}, count: {count}, y_true: {len(y_true)}, rate: {count / len(y_true) * 100}")
    print(f"k: {k}, rate: {count / len(y_true) * 100}")
        
        #print(f"pred_emotions after: {pred_emotions}")
        #print("-------------------------")
        #y_pred.append(pred_emotions)
    #print(f"true_emo_tst: {true_emo_tst[:10]}")
#    used_labels = list(range(32))
#    print(f"length: {len(y_true)}, {len(y_pred)}")
#    #program_acc = top_k_accuracy_score(y_true, y_pred, labels=used_labels, k=k)
#    print(f"k: {k}, program_acc: {program_acc}")

def load():
    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset()
    emo_prob_path = f"{config.ed_dir}/{config.emo_prob_name}"
    print(f"Loading Emotion with probability data from path: {emo_prob_path}")
    if not os.path.exists(emo_prob_path):
        print(f"File is not exists: {emo_prob_path}")
        emo_prob_gen.generate_probabilities(None, None, None)
    [emo_tra, emo_val, emo_tst] = load_json(emo_prob_path)
    return emo_tst, pairs_tst

def run():
    emo_tst, pairs_tst = load()
    check_list = [1, 2, 3, 4, 5, 10, 15, 20, 25, 32]
    #check_list = [1]
    for k in check_list:
        test(k, emo_tst, pairs_tst)

run()
