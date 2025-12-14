"""
Evaluation module - evaluate model performance based on generated outputs.
The evaluation logic is refactored and organized while keeping metrics identical to
the original implementation.
"""
import json
import datetime
import os
import sys
import re
from typing import Tuple, List, Optional, Dict

# Add path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import f1_score, classification_report
try:
    from nltk.tokenize import word_tokenize
except ImportError:
    # Fallback to simple whitespace tokenization if nltk is not available
    def word_tokenize(text):
        return text.split()


def calc_distinct_n(n: int, candidates: List[str], print_score: bool = True) -> float:
    """
    Calculate distinct-n score for a list of candidate texts.
    
    Args:
        n: n-gram size (1 for unigrams, 2 for bigrams, etc.)
        candidates: List of candidate text strings
        print_score: Whether to print the score
    
    Returns:
        Distinct-n score (ratio of unique n-grams to total n-grams)
    """
    ngram_dict = {}
    total = 0
    tokenized_candidates = [word_tokenize(candidate) for candidate in candidates]
    for sentence in tokenized_candidates:
        for i in range(len(sentence) - n + 1):
            ngram = tuple(sentence[i : i + n])
            ngram_dict[ngram] = 1
            total += 1
    score = len(ngram_dict) / (total + 1e-16)

    if print_score:
        print(f"***** Distinct-{n}: {score*100} *****")

    return score


def calc_distinct(candidates: List[str], print_score: bool = True) -> Tuple[float, float]:
    """
    Calculate distinct-1 and distinct-2 scores for a list of candidate texts.
    
    Args:
        candidates: List of candidate text strings
        print_score: Whether to print the scores
    
    Returns:
        Tuple of (distinct_1, distinct_2) scores
    """
    scores = []
    for i in range(2):
        score = calc_distinct_n(i + 1, candidates, print_score)
        scores.append(score)
    return scores[0], scores[1]


class Evaluator:
    """Evaluator responsible for assessing model outputs."""
    
    def __init__(self, auxiliary_model: str, data_type: str):
        self.auxiliary_model = auxiliary_model.upper()
        self.data_type = data_type.lower()
        self.label_dict, self.emo_set = self._build_label_dict()
    
    def _build_label_dict(self) -> Tuple[dict, List[str]]:
        """Build mapping from emotion string to label index."""
        if self.auxiliary_model == "EI":
            if self.data_type == "ed":
                origin_emotion_text = "surprised,excited,annoyed,proud,angry,sad,grateful,lonely,impressed,afraid,disgusted,confident,terrified,hopeful,anxious,disappointed,joyful,prepared,guilty,furious,nostalgic,jealous,anticipating,embarrassed,content,devastated,sentimental,caring,trusting,ashamed,apprehensive,faithful"
            elif self.data_type == "edos":
                origin_emotion_text = "afraid,agreeing,angry,annoyed,anticipating,anxious,apprehensive,ashamed,caring,confident,consoling,content,devastated,disappointed,disgusted,embarrassed,encouraging,excited,faithful,furious,grateful,guilty,hopeful,impressed,jealous,joyful,lonely,neutral,nostalgic,prepared,proud,questioning,sad,sentimental,suggesting,surprised,sympathizing,terrified,trusting,wishing"
            elif self.data_type == "ge":
                origin_emotion_text = "afraid,angry,annoyed,anxious,caring,disappointed,disgusted,embarrassed,excited,grateful,guilty,hopeful,impressed,joyful,neutral,proud,sad,surprised,wishing"
        elif self.auxiliary_model == "GE":
            if self.data_type == "ed":
                origin_emotion_text = "caring,admiration,anger,annoyance,disappointment,disgust,embarrassment,excitement,fear,gratitude,joy,nervousness,optimism,pride,remorse,sadness,surprise"
            elif self.data_type == "edos":
                origin_emotion_text = "caring,admiration,anger,annoyance,desire,disappointment,disgust,embarrassment,excitement,fear,gratitude,joy,nervousness,optimism,pride,remorse,sadness,surprise,neutral"
            elif self.data_type == "ei":
                origin_emotion_text = "caring,admiration,anger,annoyance,desire,disappointment,disgust,embarrassment,excitement,fear,gratitude,joy,nervousness,optimism,pride,remorse,sadness,surprise,neutral"
        
        emo_set = origin_emotion_text.lower().split(",")
        label_dict = dict([(emo, i) for (i, emo) in enumerate(emo_set)])
        return label_dict, emo_set
    
    def _sentiment2label(self, text: str) -> int:
        """Convert emotion text into a label ID."""
        for key in self.label_dict:
            if key in text.lower():
                return self.label_dict[key]
        return 1

    def _extract_emotion(self, text: str) -> str:
        """Extract emotion token like 'guilty' from text containing 'Emotion:guilty'."""
        if not text:
            return ""
        # regex first
        m = re.search(r"emotion[:：]\s*([a-zA-Z0-9_\-]+)", text, re.IGNORECASE)
        if m:
            return m.group(1).lower()
        # fallback: try to find any known emotion substring
        lower_text = text.lower()
        for key in self.emo_set:
            if key in lower_text:
                return key
        return ""
    
    def _read_jsonl_file(self, file_path: str) -> Tuple[float, float, List[str], float, List[int], List[int]]:
        """
        Read a jsonl file and compute intermediate statistics.

        Returns:
            (emo_count, emo_total, response_data, macro_f1, true_labels, pred_labels)
        """
        res_data = []
        emo_count = 0.0
        emo_total = 0.0
        true_labels = []
        pred_labels = []
        
        with open(file_path, 'r') as file:
            for line in file:
                line_json = json.loads(line)
                label_text = line_json.get("label", "")
                predict_text = line_json.get("predict", "")

                emotion = self._extract_emotion(label_text)
                pred_emotion = self._extract_emotion(predict_text)
                #import ipdb; ipdb.set_trace()

                # fallback to original slicing if regex failed
                if not emotion:
                    emotion = label_text.split("\n ")[0].replace("Emotion:", "").replace(" ", "")

                pred_response = predict_text
                
                true_label = self._sentiment2label(emotion)
                pred_label = self._sentiment2label(pred_emotion)
                
                true_labels.append(true_label)
                pred_labels.append(pred_label)
                emo_total += 1.0
                
                if emotion.lower() in pred_emotion.lower():
                    emo_count += 1.0
                
                res_data.append(pred_response)
        
        macro_f1 = f1_score(true_labels, pred_labels, average='macro') * 100
        print(classification_report(true_labels, pred_labels))
        
        return emo_count, emo_total, res_data, macro_f1, true_labels, pred_labels
    
    def evaluate(self, file_path: str, exp_id: str = "") -> Dict:
        """
        Evaluate model outputs stored in a jsonl file.

        Args:
            file_path: Path to the jsonl output file.
            exp_id: Optional experiment identifier.

        Returns:
            Dictionary with keys:
                'true_labels': List[int],
                'pred_labels': List[int],
                'emo_set': List[str],
                'accuracy': float,
                'macro_f1': float,
                'distinct_1': float,
                'distinct_2': float,
                'num_correct': float,
                'num_samples': float
        """
        if not os.path.exists(file_path):
            print(f"Error: evaluation file does not exist: {file_path}")
            return {
                'true_labels': [],
                'pred_labels': [],
                'emo_set': [],
                'accuracy': 0.0,
                'macro_f1': 0.0,
                'distinct_1': 0.0,
                'distinct_2': 0.0,
                'num_correct': 0.0,
                'num_samples': 0.0,
            }
        
        print(f"Start evaluation: {file_path}")
        emo_count, emo_total, response_data, macro_f1, true_labels, pred_labels = \
            self._read_jsonl_file(file_path)
        
        emo_rate = emo_count / emo_total * 100
        print(f"emo_count: {emo_count}, emo_total: {emo_total}, "
              f"rate: {emo_rate:.2f}%, macro_f1: {macro_f1:.2f}")
        
        # Calculate Distinct metrics
        d1, d2 = calc_distinct(response_data)
        print(f"Dist-1: {d1:.4f}, Dist-2: {d2:.4f}")
        
        # Save evaluation summary to a text file
        base_dir = "output/"
        out_path = os.path.join(base_dir, 'ER_result.txt')
        current_time = datetime.datetime.now()
        out_txt = (
            f"{current_time}\t{exp_id}\t[Acc]\t[Macro_f1]\t[Correct]\t[Total]\n"
            f"{emo_rate:.3f}, {macro_f1:.3f}, {emo_count:.3f}, {emo_total:.3f}\n\n"
        )
        
        os.makedirs(base_dir, exist_ok=True)
        with open(out_path, "a") as file:
            file.write(out_txt)
        
        return {
            'true_labels': true_labels,
            'pred_labels': pred_labels,
            'emo_set': self.emo_set,
            'accuracy': emo_rate,
            'macro_f1': macro_f1,
            'distinct_1': d1,
            'distinct_2': d2,
            'num_correct': emo_count,
            'num_samples': emo_total,
        }

