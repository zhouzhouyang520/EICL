import argparse
import os
import pickle
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

from data_processor.utils.loader import load_dataset, save_data
from data_processor.utils.tools.time_warpper import wrapper_calc_time
from data_processor.utils.tools.retriver import Retriver
from data_processor.sent_model import SentModel
# emotion_probability_generator will be imported lazily when needed


@dataclass
class BuildOptions:
    data_root: str
    output_root: str
    auxiliary_model: str
    data_name: str
    data_type: str
    experiment_type: str
    example_num: int
    batch_size: int
    use_gpu: bool
    device_id: str
    embed_model: Optional[str]
    regenerate_embeddings: bool
    regenerate_examples: bool
    regenerate_emo_prob: bool
    # EICL-specific parameters
    alpha: float = 0.2
    label_score_num: int = 5
    label_num: int = 2

    def __post_init__(self):
        self.auxiliary_model = self.auxiliary_model.upper()
        self.data_name = self.data_name.upper()
        self.experiment_type = self.experiment_type.upper()
        if not self.data_type:
            self.data_type = self.data_name.lower()
        else:
            self.data_type = self.data_type.lower()
        if not self.embed_model:
            self.embed_model = "emotion" if self.experiment_type == "EICL" else "semantic"

    @property
    def dataset_dir(self) -> str:
        rel = f"{self.auxiliary_model}_auxiliary_model_data/{self.data_name}"
        return os.path.join(self.data_root, rel)

    @property
    def embed_cache_path(self) -> str:
        fname = "dataset_emo_emb.p" if self.embed_model == "emotion" else "dataset_emb.p"
        return os.path.join(self.dataset_dir, fname)

    @property
    def example_cache_path(self) -> str:
        fname = "example_emo_index.p" if self.embed_model == "emotion" else "example_index.p"
        return os.path.join(self.dataset_dir, fname)

    @property
    def emo_prob_path(self) -> str:
        return os.path.join(self.dataset_dir, "emo_prob.json")

    @property
    def output_dir(self) -> str:
        """
        Directory to store generated json files.

        Align the directory structure with the runtime reader (`DataBuilder.get_data_path`),
        which uses:
          - ""          for ZERO-SHOT
          - "baseline" for BASELINE
          - UPPERCASE  for other experiment types (ICL, EICL, ...)
        """
        exp_type = self.experiment_type  # already upper-cased in __post_init__
        if exp_type == "ZERO-SHOT":
            exp_path = ""
        elif exp_type == "BASELINE":
            exp_path = "baseline"
        else:
            exp_path = exp_type
        return os.path.join(self.output_root, exp_path, self.auxiliary_model)

    @property
    def output_path(self) -> str:
        file_name = f"{self.data_type}_tst.json"
        return os.path.join(self.output_dir, file_name)


def parse_args() -> BuildOptions:
    parser = argparse.ArgumentParser(description="Build json data for EICL/ICL/baseline experiments.")
    parser.add_argument("--data_root", type=str, default="data", help="Root directory that stores auxiliary model data.")
    parser.add_argument("--output_root", type=str, default="data/json_data", help="Directory to store generated json files.")
    parser.add_argument("--auxiliary_model", type=str, default="EI", help="Auxiliary model name, e.g. EI or GE.")
    parser.add_argument("--data_name", type=str, default="ED", help="Dataset name, e.g. ED, EDOS, GE.")
    parser.add_argument("--data_type", type=str, default="", help="Output file prefix (defaults to lower-case data_name).")
    parser.add_argument("--experiment_type", type=str, choices=["baseline", "ICL", "EICL"], default="ICL")
    parser.add_argument("--example_num", type=int, default=5, help="Number of retrieved examples to include.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for embedding models.")
    parser.add_argument("--device_id", type=str, default="0")
    parser.add_argument("--embed_model", type=str, choices=["semantic", "emotion"], default=None, help="Embedding model type.")
    parser.add_argument("--regenerate_embeddings", action="store_true", help="Force re-building sentence embeddings.")
    parser.add_argument("--regenerate_examples", action="store_true", help="Force re-building retrieval indices.")
    parser.add_argument("--regenerate_emo_prob", action="store_true", help="Force re-computing RoBERTa emotion scores.")
    parser.add_argument("--alpha", type=float, default=0.2, help="Alpha parameter for EICL emotion construction.")
    parser.add_argument("--label_score_num", type=int, default=5, help="Label score number for format_example in EICL.")
    parser.add_argument("--label_num", type=int, default=2, help="Label number for format_context in EICL.")
    return BuildOptions(**vars(parser.parse_args()))


def configure_device(opts: BuildOptions):
    if opts.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opts.device_id)
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def link_sentence(sentences: Iterable[Iterable[str]]) -> str:
    joined = []
    for sentence in sentences:
        if isinstance(sentence, str):
            joined.append(sentence)
        else:
            joined.append(" ".join(sentence))
    return "</s>".join(joined)


def preprocess_split(split: Optional[dict]):
    if not split:
        return
    for idx, ctx in enumerate(split["context"]):
        if isinstance(ctx, str):
            continue
        split["context"][idx] = link_sentence(ctx)


def generate_embed(data_split: Optional[dict], model, batch_size: int) -> List[List[float]]:
    if not data_split:
        return []
    contexts = data_split["context"]
    result = []
    batch = []
    for ctx in tqdm(contexts, desc="Encoding sentences", leave=False):
        batch.append(link_sentence(ctx))
        if len(batch) >= batch_size:
            result.extend(model.gen_emb(batch))
            batch = []
    if batch:
        result.extend(model.gen_emb(batch))
    return result


def ensure_embeddings(opts: BuildOptions, train_data: dict, test_data: dict):
    cache_path = opts.embed_cache_path
    if not opts.regenerate_embeddings and os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        with open(cache_path, "rb") as f:
            loaded = pickle.load(f)
        # Backward compatibility: cached file may contain [train, dev, test] or [train, test].
        if isinstance(loaded, (list, tuple)):
            if len(loaded) == 3:
                emb_train, _, emb_test = loaded
            elif len(loaded) == 2:
                emb_train, emb_test = loaded
            else:
                raise ValueError(f"Unexpected embedding cache format with {len(loaded)} elements.")
            return emb_train, emb_test
        raise ValueError("Unexpected embedding cache format.")

    print("Generating sentence embeddings (train & test only)...")
    model = SentModel(opts.use_gpu, opts.device_id)
    emb_train = generate_embed(train_data, model, opts.batch_size)
    emb_test = generate_embed(test_data, model, opts.batch_size)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        # Store only what we actually use: train & test embeddings.
        pickle.dump([emb_train, emb_test], f)
    print(f"Saved embeddings to {cache_path}")
    return emb_train, emb_test


def build_example_index(train_vectors, test_vectors, topk: int = 50):
    retriver = Retriver(topk=topk, train_vectors=train_vectors, test_vectors=test_vectors)
    return retriver.search(test_vectors)


def ensure_example_index(opts: BuildOptions, train_vectors, test_vectors):
    cache_path = opts.example_cache_path
    if not opts.regenerate_examples and os.path.exists(cache_path):
        print(f"Loading cached example index from {cache_path}")
        with open(cache_path, "rb") as f:
            example_index = pickle.load(f)
        example_len = len(example_index[0][0][0])
        if example_len >= opts.example_num:
            return example_index
        print(f"Cached index has {example_len} examples < requested {opts.example_num}, rebuilding.")

    example_index = build_example_index(train_vectors, test_vectors)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(example_index, f)
    print(f"Saved example index to {cache_path}")
    return example_index


def ensure_emo_probabilities(opts: BuildOptions, train_data, test_data):
    cache_path = opts.emo_prob_path
    if not opts.regenerate_emo_prob and os.path.exists(cache_path):
        print(f"Loading cached emotion probabilities from {cache_path}")
        with open(cache_path, "rb") as f:
            loaded = pickle.load(f)
        # Backward compatibility: cached file may contain [train, dev, test] or [train, test].
        if isinstance(loaded, (list, tuple)):
            if len(loaded) == 3:
                emo_train, _, emo_test = loaded
            elif len(loaded) == 2:
                emo_train, emo_test = loaded
            else:
                raise ValueError(f"Unexpected emotion probability cache format with {len(loaded)} elements.")
            return emo_train, emo_test
        raise ValueError("Unexpected emotion probability cache format.")

    # Lazy import emotion_probability_generator
    from data_processor import emotion_probability_generator as emo_prob_gen
    # We only care about train & test splits here; pass None for dev_data.
    emo_scores = emo_prob_gen.generate_probabilities(
        train_data,
        None,
        test_data,
        batch_size=opts.batch_size,
        use_gpu=opts.use_gpu,
        device_id=opts.device_id,
    )
    # ``generate_probabilities`` may still return three splits; keep only train/test.
    if isinstance(emo_scores, (list, tuple)):
        if len(emo_scores) == 3:
            emo_train, _, emo_test = emo_scores
        elif len(emo_scores) == 2:
            emo_train, emo_test = emo_scores
        else:
            raise ValueError(f"Unexpected emotion probability result format with {len(emo_scores)} elements.")
    else:
        raise ValueError("Emotion probability generator must return a sequence.")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        # Store only what we actually use: train & test emotion scores.
        pickle.dump([emo_train, emo_test], f)
    print(f"Saved emotion probabilities to {cache_path}")
    return emo_train, emo_test


def construct_emotions(pred_emotions: Sequence[str], true_emotion: Optional[str], acc_num: int, alpha: float = 0.2) -> str:
    emotion_candidates = []
    valid_probs = []

    for emo_prob in pred_emotions[:acc_num]:
        emotion, prob = emo_prob.replace("(", "").replace(")", "").split(",")
        emotion_candidates.append(emotion)
        float_prob = round(float(prob) * alpha, 2)
        valid_probs.append(float_prob)

    if true_emotion is None:
        return ",".join(emotion_candidates)

    label_prob = round(1.0 - round(sum(valid_probs), 2), 2)
    if true_emotion in emotion_candidates:
        label_index = emotion_candidates.index(true_emotion)
        valid_probs[label_index] = round(valid_probs[label_index] + label_prob, 2)
    else:
        emotion_candidates.insert(0, true_emotion)
        valid_probs.insert(0, label_prob)
        emotion_candidates = emotion_candidates[:acc_num]
        valid_probs = valid_probs[:acc_num]

    zipped = zip(emotion_candidates, valid_probs)
    return ",".join(f"({emo},{prob})" for emo, prob in zipped)


def format_example(train_data, idx: int, opts: BuildOptions, emo_scores=None) -> str:
    context = train_data["context"][idx]
    emotion = train_data["emotion"][idx].lower()
    label_score_num = opts.label_score_num
    if opts.experiment_type == "EICL" and emo_scores is not None:
        emo_string = construct_emotions(emo_scores[idx], emotion, label_score_num, opts.alpha)
        return f"Dialogue example: {context}\tPredicted emotion and probability: {emo_string}"
    return f"Dialogue example: {context}\tTrue emotion label: {emotion}"


def format_context(test_data, idx: int, opts: BuildOptions, emo_scores=None) -> str:
    context = test_data["context"][idx]
    label_num = opts.label_num
    if opts.experiment_type == "EICL" and emo_scores is not None:
        emo_string = construct_emotions(emo_scores[idx], None, label_num, opts.alpha)
        return f"{emo_string}\nDialogue context: {context}"
    return f"Dialogue context: {context}"


def format_output(test_data, idx: int, opts: BuildOptions) -> str:
    emotion = test_data["emotion"][idx]
    if len(test_data["target"]) > 0:
        target_words = test_data["target"][idx]
        response = " ".join(target_words)
        if opts.experiment_type == "EICL":
            return f"Emotion:{emotion}\n Response:{response}"
        return f"Emotion:{emotion}\n Response:{response}"
    else:
        return f"Emotion:{emotion}"


def build_examples_block(example_index, sample_idx: int, opts: BuildOptions, train_data, emo_scores):
    if example_index is None:
        return ""
    indices, _ = example_index[sample_idx]
    indices = getattr(indices, "tolist", lambda: indices)()
    if isinstance(indices[0], (list, tuple)):
        indices = indices[0]
    selected = indices[:opts.example_num]
    examples = []
    for id_ in selected:
        if id_ < 0 or id_ >= len(train_data["context"]):
            continue
        examples.append(format_example(train_data, id_, opts, emo_scores))
    return "\n".join(examples)


def build_records(opts: BuildOptions, example_index, train_data, test_data, emo_train=None, emo_test=None):
    records = []
    total_samples = len(test_data["context"])
    for idx in range(total_samples):
        record = {"instruction": ""}
        record["examples"] = build_examples_block(example_index, idx, opts, train_data, emo_train)
        record["input"] = format_context(test_data, idx, opts, emo_test)
        record["output"] = format_output(test_data, idx, opts)
        record["emotion"] = test_data["emotion"][idx]
        records.append(record)
    return records


@wrapper_calc_time()
def run(opts: BuildOptions):
    configure_device(opts)
    splits = load_dataset(opts.auxiliary_model, opts.data_name, opts.data_root)

    # Support datasets with or without a dev/validation split.
    if isinstance(splits, (list, tuple)):
        if len(splits) == 3:
            # Ignore dev split even if it is available; we only use train & test here.
            train_data, _, test_data = splits
        elif len(splits) == 2:
            train_data, test_data = splits
        else:
            raise ValueError(f"Expected 2 or 3 splits (train[, dev], test), got {len(splits)}")
    else:
        raise ValueError("load_dataset must return a sequence of splits.")

    if train_data is None or test_data is None:
        raise ValueError("Train and test splits are required.")

    example_index = None
    emo_train = emo_test = None
    if opts.experiment_type != "BASELINE":
        emb_train, emb_test = ensure_embeddings(opts, train_data, test_data)
        example_index = ensure_example_index(opts, emb_train, emb_test)
    if opts.experiment_type == "EICL":
        emo_train, emo_test = ensure_emo_probabilities(opts, train_data, test_data)

    preprocess_split(train_data)
    preprocess_split(test_data)

    records = build_records(opts, example_index, train_data, test_data, emo_train, emo_test)
    os.makedirs(opts.output_dir, exist_ok=True)
    save_data(records, opts.output_path)
    print(f"Saved {len(records)} records to {opts.output_path}")


if __name__ == "__main__":
    options = parse_args()
    run(options)
