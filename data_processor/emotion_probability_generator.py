from typing import List, Optional, Sequence

from tqdm import tqdm

from data_processor.emo_model import EmoModel

LABELS = [
    "afraid",
    "angry",
    "annoyed",
    "anticipating",
    "anxious",
    "apprehensive",
    "ashamed",
    "caring",
    "confident",
    "content",
    "devastated",
    "disappointed",
    "disgusted",
    "embarrassed",
    "excited",
    "faithful",
    "furious",
    "grateful",
    "guilty",
    "hopeful",
    "impressed",
    "jealous",
    "joyful",
    "lonely",
    "nostalgic",
    "prepared",
    "proud",
    "sad",
    "sentimental",
    "surprised",
    "terrified",
    "trusting",
    "agreeing",
    "acknowledging",
    "encouraging",
    "consoling",
    "sympathizing",
    "suggesting",
    "questioning",
    "wishing",
    "neutral",
]


def _link_sentence(context) -> str:
    if isinstance(context, str):
        return context.replace("\n", "</s>")
    joined = [" ".join(sentence) for sentence in context]
    return "</s>".join(joined)


def _format_probabilities(logits: Sequence[Sequence[float]]) -> List[List[str]]:
    formatted = []
    for sample in logits:
        emo_prob = list(zip(LABELS, sample))
        emo_prob.sort(key=lambda item: item[1], reverse=True)
        formatted.append([f"({emo},{round(prob, 2)})" for emo, prob in emo_prob])
    return formatted


def _generate_for_split(dataset: Optional[dict], model: EmoModel, batch_size: int) -> List[List[str]]:
    if not dataset:
        return []
    contexts = [_link_sentence(ctx) for ctx in dataset["context"]]
    scores = []
    for start in tqdm(range(0, len(contexts), batch_size), desc="Scoring emotions", leave=False):
        batch = contexts[start : start + batch_size]
        logits = model.gen_emb(batch, mode="EDOS")
        scores.extend(_format_probabilities(logits))
    return scores


def generate_probabilities(
    train_data: dict,
    dev_data: Optional[dict],
    test_data: dict,
    batch_size: int = 16,
    use_gpu: bool = False,
    device_id: str = "0",
):
    """Generate emotion probabilities for each split using EmoBERT."""
    _ = use_gpu  # Placeholder: EmoModel manages device placement internally.
    model = EmoModel(device_id=device_id)
    emo_train = _generate_for_split(train_data, model, batch_size)
    emo_dev = _generate_for_split(dev_data, model, batch_size)
    emo_test = _generate_for_split(test_data, model, batch_size)
    return [emo_train, emo_dev, emo_test]
