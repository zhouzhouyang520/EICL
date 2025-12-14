"""
Local small-scale model runner - simplified version.
Core logic based on honesty_prob.py, removing rep_dict and other unrelated code.
Uses Transformers directly for generation.
"""
import torch
import json
import os
import sys
import jsonlines
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from modules.evaluator import Evaluator
from modules.prompt_builder import (
    get_emotion_labels,
    build_instruction,
    build_all_inputs,
)


def save_data(json_data, path):
    """Save data in jsonl format."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode='w') as file:
        writer = jsonlines.Writer(file)
        for item in json_data:
            writer.write(item)


def run_local_model(config) -> tuple:
    """
    Run local small-scale model.
    
    Args:
        config: Config object containing all configuration info.
    
    Returns:
        (output_path, eval_result_dict)
    """
    # Build experiment_id
    exp_id = f"{config.data.data_type}_{config.model.model_name}_{config.experiment.experiment_type}_{config.data.auxiliary_model}"
    
    print(f"Starting local model run: {exp_id}")
    
    # 0. Clear GPU cache before loading model to ensure clean state
    if torch.cuda.is_available():
        print("Clearing GPU cache before model loading...")
        torch.cuda.empty_cache()
        # Get GPU memory info
        if torch.cuda.device_count() > 0:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3
            gpu_reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"GPU memory: {gpu_memory:.2f} GB total, {gpu_allocated:.2f} GB allocated, {gpu_reserved:.2f} GB reserved")
    
    # 1. Load model and tokenizer (use get_model_path for correct mapping)
    model_path = config.model.get_model_path()
    print(f"Loading model: {model_path}")
    
    # Add model path to sys.path for transformers to find custom code
    abs_model_path = os.path.abspath(model_path)
    if abs_model_path not in sys.path:
        sys.path.insert(0, abs_model_path)
    
    # Determine device: prefer GPU if available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Target device: {device}")
    
    # Transformers 4.45.0 natively supports Phi3, load directly
    # Use device_map="auto" which should prefer GPU, then explicitly move if needed
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        device_map="auto",
        trust_remote_code=True  # Allow loading custom code if needed
    )
    
    # Verify and ensure model is on the correct device
    model_device = next(model.parameters()).device
    print(f"Model initially loaded on device: {model_device}")
    
    if torch.cuda.is_available():
        # Check if model is on CPU when GPU is available
        if "cpu" in str(model_device):
            print("WARNING: Model loaded on CPU despite GPU availability! This may cause slow performance.")
            print("Attempting to move model to GPU...")
            # For models with device_map, we need to handle differently
            # Try to move the model to GPU
            try:
                model = model.to(device)
                torch.cuda.empty_cache()
                model_device = next(model.parameters()).device
                print(f"Model moved to device: {model_device}")
            except Exception as e:
                print(f"Failed to move model to GPU: {e}")
                print("Model will run on CPU (this will be slow)")
        else:
            print(f"Model is correctly on GPU: {model_device}")
            # Log GPU memory after loading
            if torch.cuda.is_available():
                gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3
                gpu_reserved = torch.cuda.memory_reserved(0) / 1024**3
                print(f"GPU memory after model load: {gpu_allocated:.2f} GB allocated, {gpu_reserved:.2f} GB reserved")
    else:
        print("No GPU available, model running on CPU")
    
    use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        use_fast=use_fast_tokenizer, 
        padding_side="left", 
        legacy=False,
        trust_remote_code=True  # Allow loading custom tokenizer code
    )
    tokenizer.pad_token_id = 0
    
    user_tag = "[INST]"
    assistant_tag = "[/INST]"
    
    # 2. Load test data
    data_path = config.data.runtime_data_path
    # Use uppercase to match DataBuilder
    exp_type = config.experiment.experiment_type.upper()
    if exp_type == "ZERO-SHOT":
        exp_type_path = ""
    elif exp_type == "BASELINE":
        exp_type_path = "baseline"
    else:
        exp_type_path = exp_type
    
    if exp_type == "EICL":
        data_file = f"{config.data.data_type}_{config.model.model_name}_{config.experiment.experiment_type}_{config.data.auxiliary_model}_data.json"
    else:
        data_file = f"{config.data.data_type}_{config.experiment.experiment_type}_{config.data.auxiliary_model}_data.json"
    test_json_path = os.path.join(data_path, exp_type_path, config.data.auxiliary_model, data_file)
    print(f"Loading test data: {test_json_path}")
    
    # Check if file exists
    if not os.path.exists(test_json_path):
        error_msg = f"\nError: test data file not found: {test_json_path}\n"
        error_msg += f"Please ensure data has been built.\n"
        error_msg += f"Data path structure: {data_path}/{exp_type_path}/{config.data.auxiliary_model}/{data_file}\n"
        error_msg += f"Note: main.py will build data automatically, or use DataBuilder manually.\n"
        raise FileNotFoundError(error_msg)
    
    with open(test_json_path, "r", encoding="utf-8") as f:
        test_json = json.load(f)
    
    # 3. Get emotion labels and instruction
    origin_emotion_text = get_emotion_labels(config.data.auxiliary_model, config.data.data_type)
    total_emotions = origin_emotion_text.split(",")
    # build_instruction expects "baseline", "ICL", "EICL" format
    instruct_str = build_instruction(config.experiment.experiment_type, origin_emotion_text)
    
    # 4. Build all inputs (in test_json order)
    all_inputs, all_origins = build_all_inputs(
        test_json=test_json,
        config=config,
        user_tag=user_tag,
        assistant_tag=assistant_tag,
    )
    print(f"Preparing to process {len(all_inputs)} samples")
    
    # 5. Generate results
    # Use the generate_batch_size resolved in Config (supports per-model/exp overrides)
    generate_batch_size = config.model.generate_batch_size
    # Ensure runtime and generate complete prediction results.
    if "llama" in config.model.model_name.lower():
        max_new_tokens = 50 # To reduce runtime
    else:
        max_new_tokens = 10 # To reduce runtime
    
    update_generation = []
    
    print("Starting prediction generation...")
    
    # Batch generation
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(all_inputs), generate_batch_size), desc="Generating"):
            batch_inputs = all_inputs[i:i + generate_batch_size]
            batch_origins = all_origins[i:i + generate_batch_size]
            
            # Tokenize inputs
            encodings = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True)
            encodings = {k: v.to(model.device) for k, v in encodings.items()}
            
            #import ipdb; ipdb.set_trace()
            # Generate
            outputs = model.generate(
                **encodings,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id,
            )
            
            input_ids = encodings["input_ids"]
            attention_mask = encodings["attention_mask"]
            
            # Decode outputs – remove the input prompt using the true token length
            for j, (origin, output_ids) in enumerate(zip(batch_origins, outputs)):
                # Effective length of the input sequence (non-padding tokens)
                generated_text = tokenizer.decode(output_ids, skip_special_tokens=False).split(assistant_tag)[1]

                # Build output dict (format matches original code)
                output_label = origin.get("output", "")  # Original output field
                sample_idx = origin.get("sample_idx", 0)
                avg_weight = origin.get("avg_weight", 0.0)
                
                update_dict = {
                    "label": output_label,
                    "predict": generated_text,
                    "sample_idx": sample_idx,
                    "avg_weight": avg_weight
                }
                update_generation.append(update_dict)
    
    # 6. Save results
    output_path = os.path.join(
        config.output.output_base_path,
        exp_type_path if exp_type_path else "",
        config.data.auxiliary_model,
        f"{exp_id}_generated_predictions.jsonl"
    )
    
    # Sort by sample_idx
    update_generation.sort(key=lambda x: x.get('sample_idx', 0))
    save_data(update_generation, output_path)
    print(f"Results saved to: {output_path}")
    
    # 7. Evaluate results
    print("Starting evaluation...")
    evaluator = Evaluator(
        auxiliary_model=config.data.auxiliary_model,
        data_type=config.data.data_type
    )
    eval_result = evaluator.evaluate(output_path, exp_id)
    
    # 8. Clean up: delete model and clear GPU cache for next experiment
    print("Cleaning up model and GPU memory...")
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print("GPU cache cleared")
    
    return output_path, eval_result
