"""
Remote model runner - handles API requests for ChatGPT and Claude models.
Uses requests library for API calls (from api_utils.py) and 
data processing logic.
"""
import json
import os
import time
import jsonlines
import requests
import threading
from typing import Optional, Tuple, List, Dict
from math import ceil
from tqdm import tqdm

from modules.evaluator import Evaluator
# Reuse configuration and request building logic from api_utils to ensure consistency
from modules.api_utils import (
    load_config as load_api_test_config,
    build_request as build_api_request,
    parse_response as parse_api_response,
)
from modules.prompt_builder import build_all_inputs


def save_data(json_data, path):
    """Save data in jsonl format, filtering None values and sorting by index."""
    json_data = filter(lambda x: x is not None, json_data)
    json_data = list(sorted(json_data, key=lambda x: x.get("index", 0) if isinstance(x, dict) else 0))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode='w') as file:
        writer = jsonlines.Writer(file)
        for item in json_data:
            writer.write(item)


class RemoteApiClient:
    """API client for remote model requests using requests library."""

    def __init__(self, provider: str, api_cfg: Dict[str, str]):
        """
        Initialize API client.

        Args:
            provider: 'gpt' or 'claude'
            api_cfg: config for provider (url/api_key/model)
        """
        self.provider = provider
        self.api_cfg = api_cfg

    def request_data(self, content: str, timeout: int = 30) -> Tuple[Optional[str], int]:
        """
        Send API request with given content.

        Args:
            content: Input text content
            timeout: Request timeout in seconds

        Returns:
            (response_content, status_code)
        """
        request_data = build_api_request(self.provider, self.api_cfg, content)

        try:
            response = requests.post(
                request_data["url"],
                headers=request_data["headers"],
                json=request_data["json"],
                timeout=timeout,
            )

            if response.status_code == 200:
                res = parse_api_response(self.provider, response)
                return res, response.status_code
            else:
                # log body for debugging
                print(f"[API][{self.provider}] status={response.status_code} body={response.text}")
                return None, response.status_code

        except Exception as e:
            print(f"API request error: {e}")
            return None, 500


def chat_test_data(
    api_client: RemoteApiClient,
    prepared_inputs: List[str],
    origins: List[Dict],
    test_flag: bool,
    results: List,
    start: int,
    end: int
) -> List:
    """
    Process test data with API calls (multi-threaded worker function).
    
    Args:
        api_client: RemoteApiClient instance
        instruction: Instruction text
        test_data: List of test data dictionaries
        test_flag: Whether in test mode (prints debug info)
        results: Result list to append to
        start: Start index in test_data
        end: End index in test_data
    
    Returns:
        Updated results list
    """
    retries = 5
    current_index = start
    
    for idx in tqdm(range(start, end), desc=f"Processing {start}-{end}"):
        data = origins[idx]
        one_data = {}
        output_data = data.get("output", "")
        full_input = prepared_inputs[idx]
        
        if test_flag:
            print(f"input_data: {full_input}")
            print("=============================")
        
        chat_out = None
        status_code = 200
        
        # Retry logic with exponential backoff
        for attempt in range(retries):
            try:
                chat_out, status_code = api_client.request_data(full_input)
            except Exception as e:
                if attempt < retries - 1:
                    sleep_time = min(2 ** attempt, 8)  # Cap at 8 seconds
                    time.sleep(sleep_time)
                    continue
                else:
                    chat_out = None
                    status_code = 500
            
            if status_code == 200 and chat_out is not None:
                break
            else:
                # brief debug info per attempt
                print(f"[API][retry={attempt}] status={status_code}, empty_response={chat_out is None}")
        
        # Build result entry
        one_data["label"] = "[gMASK]sop " + output_data
        one_data["predict"] = chat_out if chat_out else ""
        one_data["index"] = current_index
        
        # Also include sample_idx and avg_weight if present
        if "sample_idx" in data:
            one_data["sample_idx"] = data["sample_idx"]
        if "avg_weight" in data:
            one_data["avg_weight"] = data["avg_weight"]
        
        results[current_index] = one_data
        current_index += 1
    
    return results


def get_default_api_config(model_name: str) -> Tuple[str, str]:
    """
    Get default API URL and key based on model name.
    
    Args:
        model_name: Model name (e.g., "ChatGPT", "Claude", "gpt-3.5-turbo")
    
    Returns:
        (api_url, api_key)
    """
    model_lower = model_name.lower()
    
    # Default API endpoint (can be overridden in config)
    if "claude" in model_lower:
        # Default Claude API endpoint
        api_url = "https://api.anthropic.com/v1/messages"
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
    else:
        # Default OpenAI-compatible API endpoint
        api_url = "https://api.jisuai.top/v1/chat/completions"
        api_key = os.getenv("OPENAI_API_KEY", "")
    
    return api_url, api_key


def get_api_model_name(model_name: str, api_model: Optional[str] = None) -> str:
    """
    Get API model name from config or infer from model_name.
    
    Args:
        model_name: Model name from config
        api_model: Explicit API model name (if set in config)
    
    Returns:
        API model name string
    """
    if api_model:
        return api_model
    
    model_lower = model_name.lower()
    if "claude" in model_lower:
        return "claude-haiku-4-5-20251001" #"claude-3-haiku-20240307"#"claude-3-haiku"
    elif "gpt" in model_lower or "chatgpt" in model_lower:
        return "gpt-4o-mini"#"gpt-3.5-turbo"
    else:
        return "gpt-4o-mini"#"gpt-3.5-turbo"  # Default fallback


def run_remote_model(config) -> Tuple[Optional[str], Optional[dict]]:
    """
    Run remote model via API (ChatGPT or Claude).
    
    Args:
        config: Config object containing all configuration info.
    
    Returns:
        (output_path, eval_result_dict) or (None, None) on failure.
    """
    # Build experiment_id
    exp_id = f"{config.data.data_type}_{config.model.model_name}_{config.experiment.experiment_type}_{config.data.auxiliary_model}"
    print(f"Starting remote model run: {exp_id}")
    
    # 1. Get API configuration (prefer api_utils config from run.json)
    provider = "claude" if "claude" in config.model.model_name.lower() else "gpt"
    api_config = load_api_test_config().get("api", {}).get(provider, {}).copy()

    # Allow overriding url/key/model from main config
    if config.model.api_url:
        api_config["url"] = config.model.api_url
    if config.model.api_key:
        api_config["api_key"] = config.model.api_key
    if config.model.api_model:
        api_config["model"] = config.model.api_model

    # Fallback to default environment variable logic if required fields are still missing
    if not api_config.get("url") or not api_config.get("api_key"):
        fallback_url, fallback_key = get_default_api_config(config.model.model_name)
        api_config.setdefault("url", fallback_url)
        api_config.setdefault("api_key", fallback_key)

    api_model_name = get_api_model_name(
        config.model.model_name,
        api_config.get("model", config.model.api_model),
    )
    api_config["model"] = api_model_name

    print(f"Using API provider: {provider}")
    print(f"Using API: {api_config.get('url')}")
    print(f"API model: {api_model_name}")
    
    # 2. Load test data
    data_path = config.data.runtime_data_path
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
    test_json_path = os.path.join(
        data_path, 
        exp_type_path, 
        config.data.auxiliary_model, 
        data_file
    )
    
    print(f"Loading test data: {test_json_path}")
    
    if not os.path.exists(test_json_path):
        error_msg = f"\nError: test data file not found: {test_json_path}\n"
        error_msg += f"Please ensure data has been built.\n"
        raise FileNotFoundError(error_msg)
    
    with open(test_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    # 3. Handle baseline/zero-shot: ensure examples field exists
    if exp_type == "BASELINE" or exp_type == "ZERO-SHOT":
        for data in test_data:
            if "examples" not in data or not data.get("examples"):
                data["examples"] = ""
    
    # 4. Build prompts in the same way as local_model_runner
    # For remote API prompts, do not wrap with [INST] [/INST]
    user_tag = ""
    assistant_tag = ""
    prepared_inputs, origins = build_all_inputs(
        test_json=test_data,
        config=config,
        user_tag=user_tag,
        assistant_tag=assistant_tag,
    )
    # Remote API prompt: strip leading/trailing whitespace
    prepared_inputs = [p.strip() for p in prepared_inputs]#[:10]
    if not prepared_inputs:
        print("No prepared inputs to send to API.")
        return None, None
    # 5. Create API client
    api_client = RemoteApiClient(provider, api_config)
    
    # 6. Multi-threaded API invocation
    num_threads = min(80, max(1, len(prepared_inputs)))
    test_flag = False
    
    result_list = [None] * len(prepared_inputs)
    threads = []
    
    chunk_size = ceil(len(prepared_inputs) / num_threads)
    print(f"Processing {len(prepared_inputs)} samples with {num_threads} threads...")
    
    for i in range(0, len(prepared_inputs), chunk_size):
        start = i
        end = min(i + chunk_size, len(prepared_inputs))
        # Create new API client instance for each thread
        thread_api_client = RemoteApiClient(provider, api_config)
        thread = threading.Thread(
            target=chat_test_data,
            args=(thread_api_client, prepared_inputs, origins, test_flag, result_list, start, end)
        )
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()
    
    # 7. Save results
    output_path = os.path.join(
        config.output.output_base_path,
        exp_type_path if exp_type_path else "",
        config.data.auxiliary_model,
        f"{exp_id}_generated_predictions.jsonl"
    )
    
    #import ipdb; ipdb.set_trace()
    save_data(result_list, output_path)
    print(f"Results saved to: {output_path}")
    
    # 8. Evaluate results
    print("Starting evaluation...")
    evaluator = Evaluator(
        auxiliary_model=config.data.auxiliary_model,
        data_type=config.data.data_type
    )
    eval_result = evaluator.evaluate(output_path, exp_id)
    
    return output_path, eval_result

