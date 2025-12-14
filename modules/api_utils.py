#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API utility module - provides API configuration loading, request building, and response parsing.
"""

import json
from pathlib import Path
from typing import Any, Dict

import requests

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "run.json"


def load_json(path: str) -> Any:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_instruction(data_path: str) -> str:
    """Read instruction file."""
    with open(data_path, 'r', encoding='utf-8') as f:
        content = f.read()
        return content


def load_config() -> Dict[str, Any]:
    """Load configuration file."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_request(provider: str, cfg: Dict[str, Any], message: str) -> Dict[str, Any]:
    """Build request headers and payload based on provider."""
    if provider == "gpt" or provider == "claude":
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg['api_key']}"
        }
        payload = {
            "model": cfg["model"],
            "messages": [{"role": "user", "content": message}]
        }
    # elif provider == "claude":
    #     headers = {
    #         "Content-Type": "application/json",
    #         "x-api-key": cfg["api_key"],
    #         "anthropic-version": "2023-06-01"
    #     }
    #     payload = {
    #         "model": cfg["model"],
    #         "max_tokens": 256,
    #         "messages": [{"role": "user", "content": message}]
    #     }
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return {"url": cfg["url"], "headers": headers, "json": payload}


def parse_response(provider: str, response: requests.Response) -> str:
    """Parse response from different providers."""
    data = response.json()
    if provider == "gpt" or provider == "claude":
        return data["choices"][0]["message"]["content"]
    #if provider == "gpt":
    #    return data["choices"][0]["message"]["content"]
    #if provider == "claude":
    #    return data["content"][0]["text"]
    raise ValueError(f"Unknown provider: {provider}")


def test_api(provider: str = "gpt", message: str = "Please reply: test successful"):
    """Test API connection."""
    print("Testing API connection...")
    config = load_config()

    if "api" not in config or provider not in config["api"]:
        raise KeyError(f"Configuration file missing API config for {provider}")

    cfg = config["api"][provider]
    request_data = build_request(provider, cfg, message)

    print(f"Provider: {provider}")
    print(f"URL: {request_data['url']}")
    print(f"Headers: {request_data['headers']}")
    print(f"Data: {json.dumps(request_data['json'], ensure_ascii=False)}")

    try:
        print("\nSending request...")
        response = requests.post(
            request_data["url"],
            headers=request_data["headers"],
            json=request_data["json"],
            timeout=30
        )

        print(f"Status code: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        print(f"Response content: {response.text}")

        if response.status_code == 200:
            content = parse_response(provider, response)
            print(f"Success! AI response: {content}")
            return True

        print(f"Failed with status code: {response.status_code}")
        return False

    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=== API Test ===")
    success = test_api()

    if success:
        print("\n✅ Test successful")
    else:
        print("\n❌ Test failed")

