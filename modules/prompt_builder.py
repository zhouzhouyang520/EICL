"""
Shared prompt construction utilities for local and remote runners.
"""
from typing import List, Tuple, Dict


def get_emotion_labels(aux_model: str, data_type: str) -> str:
    """Return emotion label list."""
    if aux_model == "EI":
        if data_type == "ed":
            return "surprised,excited,annoyed,proud,angry,sad,grateful,lonely,impressed,afraid,disgusted,confident,terrified,hopeful,anxious,disappointed,joyful,prepared,guilty,furious,nostalgic,jealous,anticipating,embarrassed,content,devastated,sentimental,caring,trusting,ashamed,apprehensive,faithful"
        if data_type == "edos":
            return "afraid,agreeing,angry,annoyed,anticipating,anxious,apprehensive,ashamed,caring,confident,consoling,content,devastated,disappointed,disgusted,embarrassed,encouraging,excited,faithful,furious,grateful,guilty,hopeful,impressed,jealous,joyful,lonely,neutral,nostalgic,prepared,proud,questioning,sad,sentimental,suggesting,surprised,sympathizing,terrified,trusting,wishing"
        if data_type == "ge":
            return "afraid,angry,annoyed,anxious,caring,disappointed,disgusted,embarrassed,excited,grateful,guilty,hopeful,impressed,joyful,neutral,proud,sad,surprised,wishing"
    elif aux_model == "GE":
        if data_type == "ed":
            return "caring,admiration,anger,annoyance,disappointment,disgust,embarrassment,excitement,fear,gratitude,joy,nervousness,optimism,pride,remorse,sadness,surprise"
        if data_type == "edos":
            return "caring,admiration,anger,annoyance,desire,disappointment,disgust,embarrassment,excitement,fear,gratitude,joy,nervousness,optimism,pride,remorse,sadness,surprise,neutral"
        if data_type == "ei":
            return "caring,admiration,anger,annoyance,desire,disappointment,disgust,embarrassment,excitement,fear,gratitude,joy,nervousness,optimism,pride,remorse,sadness,surprise,neutral"
    return ""


def build_instruction(exp_type: str, emotion_labels: str) -> str:
    """Build instruction string for prompt."""
    last_prompt = ""
    if exp_type == "baseline":
        first_prompt = "Infer the emotion of the dialogue context."
    elif exp_type == "ICL":
        first_prompt = "Based on the labeled examples provided, infer the emotion of the dialogue context.\n-The Dialogue example and True emotion label refer to the relevant dialogue and corresponding label."
    elif exp_type == "EICL":
        first_prompt = "Based on the labeled examples provided, infer the emotion of the dialogue context.\n- Dialogue example, Predicted emotion, and probability represent the relevant example, its ground truth labels, and the associated probabilities."
        last_prompt = "- Prioritize the 'More likely emotion label' during inference."
    else:
        first_prompt = "Infer the emotion of the dialogue context."

    instruct_str = f"""
{first_prompt}
- Dialogue context: The conversation history between speaker and listener, with utterances separated by </s>.
- Emotion labels: {emotion_labels}
- Choose a single inferred emotion from the provided "Emotion labels," not outside of them.
- Response Format: Emotion: [a single inferred emotion]
{last_prompt}
"""
    return instruct_str.strip()


def build_all_inputs(
    test_json: List[Dict],
    config,
    user_tag: str,
    assistant_tag: str,
) -> Tuple[List[str], List[Dict]]:
    """
    Build prompt inputs and corresponding origin data in order.

    Returns:
        (all_inputs, all_origins)
    """
    origin_emotion_text = get_emotion_labels(config.data.auxiliary_model, config.data.data_type)
    total_emotions = origin_emotion_text.split(",")
    instruct_str = build_instruction(config.experiment.experiment_type, origin_emotion_text)

    all_inputs = []
    all_origins = []
    exp_type_for_check = config.experiment.experiment_type.upper()

    for d in test_json:
        emo = d.get("emotion", "").lower()
        if emo not in total_emotions:
            continue

        emo_data = d.get("input", "")
        examples = d.get("examples", "") + "\n " if exp_type_for_check != "BASELINE" else ""
        eicl_prompt = ""
        if exp_type_for_check == "EICL":
            eicl_prompt = "\n-More likely emotion labels: "

        input_data = f"{user_tag} {instruct_str}{examples}{eicl_prompt}\n{emo_data} {assistant_tag}"
        all_inputs.append(input_data)
        all_origins.append(d)

    return all_inputs, all_origins

