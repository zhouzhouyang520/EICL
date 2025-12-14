"""
Configuration loader - load and build full configuration objects from base config files.
"""
import os
import json
from typing import Dict, Optional
from pathlib import Path

from modules.config import Config, DataConfig, ExperimentConfig, ModelConfig, OutputConfig


class ConfigLoader:
    """Configuration loader that builds full Config objects from base JSON configs."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.ei_basic_config = None
        self.ge_basic_config = None
        self.eicl_config = None
        self._load_configs()

    @staticmethod
    def _resolve_overrides(
        basic_config: Dict,
        base_key: str,
        override_key: str,
        model_name: str,
        experiment_type: str,
    ):
        """
        Resolve per-model / per-experiment overrides for numeric settings such as batch size.

        New preferred format (in config JSON):
        {
          "baseline": {"Phi-3.5": 8, "default": 4},
          "ICL": {"Phi-3.5": 8},
          "EICL": {"Phi-3.5": 6, "default": 2},
          "default": 2
        }

        Backwards compatible with the previous model_exp / experiment layout.
        """
        base_value = basic_config.get(base_key)
        overrides = basic_config.get(override_key, {})
        exp_key_upper = experiment_type.upper()
        exp_key_lower = experiment_type.lower()

        # Preferred: experiment->model mapping
        exp_section = overrides.get(exp_key_upper) or overrides.get(exp_key_lower)
        if isinstance(exp_section, dict):
            # Exact model match (case sensitive)
            if model_name in exp_section:
                return exp_section[model_name]
            # Case-insensitive model match
            for k, v in exp_section.items():
                if k.lower() == model_name.lower():
                    return v
            if "default" in exp_section:
                return exp_section["default"]

        # Global default in the new format
        if "default" in overrides and not isinstance(overrides.get("default"), dict):
            return overrides["default"]

        # Backward compatibility: model_exp based overrides
        model_over = overrides.get("model_exp", {}).get(model_name, {})
        if model_over:
            if exp_key_upper in model_over:
                return model_over[exp_key_upper]
            if exp_key_lower in model_over:
                return model_over[exp_key_lower]
            if "default" in model_over:
                return model_over["default"]

        # Backward compatibility: experiment based overrides
        exp_over = overrides.get("experiment", {})
        if exp_key_upper in exp_over:
            return exp_over[exp_key_upper]
        if exp_key_lower in exp_over:
            return exp_over[exp_key_lower]

        # Global default override for the legacy layout
        if "default" in overrides:
            return overrides["default"]

        return base_value
    
    def _load_configs(self):
        """Load all base configuration files."""
        ei_config_path = self.config_dir / "EI_basic_config.json"
        ge_config_path = self.config_dir / "GE_basic_config.json"
        eicl_config_path = self.config_dir / "EICL_config.json"
        
        if ei_config_path.exists():
            with open(ei_config_path, 'r', encoding='utf-8') as f:
                self.ei_basic_config = json.load(f)
        
        if ge_config_path.exists():
            with open(ge_config_path, 'r', encoding='utf-8') as f:
                self.ge_basic_config = json.load(f)
        
        if eicl_config_path.exists():
            with open(eicl_config_path, 'r', encoding='utf-8') as f:
                self.eicl_config = json.load(f)
    
    def get_basic_config(self, auxiliary_model: str) -> Dict:
        """Get the base configuration for a given auxiliary model."""
        if auxiliary_model.upper() == "EI":
            return self.ei_basic_config
        elif auxiliary_model.upper() == "GE":
            return self.ge_basic_config
        else:
            raise ValueError(f"Unknown auxiliary model: {auxiliary_model}")
    
    def get_emotion_labels(self, auxiliary_model: str, data_type: str) -> str:
        """Get emotion label string for a given auxiliary model and data type."""
        basic_config = self.get_basic_config(auxiliary_model)
        emotion_labels = basic_config.get("emotion_labels", {})
        return emotion_labels.get(data_type.lower(), "")
    
    def get_instruction(self, auxiliary_model: str, experiment_type: str, data_type: str) -> str:
        """Get the instruction template string."""
        basic_config = self.get_basic_config(auxiliary_model)
        templates = basic_config.get("instruction_templates", {})
        base_template = basic_config.get("instruction_base_template", "")
        
        # Get template corresponding to the experiment type
        exp_type = experiment_type.lower()
        if exp_type == "zero-shot":
            exp_type = "zero-shot"
        elif exp_type not in templates:
            exp_type = "baseline"
        
        template = templates.get(exp_type, templates.get("baseline", {}))
        first_prompt = template.get("first_prompt", "")
        last_prompt = template.get("last_prompt", "")
        
        # Get emotion labels
        emotion_labels = self.get_emotion_labels(auxiliary_model, data_type)
        
        # Format instruction
        instruction = base_template.format(
            first_prompt=first_prompt,
            last_prompt=last_prompt,
            emotion_labels=emotion_labels
        )
        
        return instruction.strip()
    
    def get_data_path(self, auxiliary_model: str, data_name: str) -> str:
        """Get input data path for a given auxiliary model and dataset name."""
        basic_config = self.get_basic_config(auxiliary_model)
        data_root = basic_config.get("data_root", "data")
        template = basic_config.get("data_path_template", "{auxiliary_model}_auxiliary_model_data/{data_name}")
        
        path = template.format(
            auxiliary_model=auxiliary_model,
            data_name=data_name
        )
        return os.path.join(data_root, path)
    
    def get_json_path(self, auxiliary_model: str, experiment_type: str, data_type: str) -> str:
        """Get JSON file path for generated data."""
        basic_config = self.get_basic_config(auxiliary_model)
        json_output_root = basic_config.get("json_output_root", "data/json_data")
        template = basic_config.get("json_path_template", "{json_output_root}/{experiment_type}/{auxiliary_model}/{data_type}_tst.json")
        
        # Handle the zero-shot case
        exp_type_path = experiment_type
        if experiment_type.lower() == "zero-shot":
            exp_type_path = ""
        elif experiment_type.lower() == "baseline":
            exp_type_path = "baseline"
        
        path = template.format(
            json_output_root=json_output_root,
            experiment_type=exp_type_path,
            auxiliary_model=auxiliary_model,
            data_type=data_type.lower()
        )
        return path
    
    def get_output_path(self, auxiliary_model: str, experiment_type: str, data_type: str, model_name: str) -> str:
        """Get output path for predictions."""
        basic_config = self.get_basic_config(auxiliary_model)
        template = basic_config.get("output_path_template", "./output/{experiment_type}/{auxiliary_model}/{data_type}_{model_name}_{experiment_type}_{auxiliary_model}_generated_predictions.jsonl")
        
        path = template.format(
            experiment_type=experiment_type,
            auxiliary_model=auxiliary_model,
            data_type=data_type.lower(),
            model_name=model_name
        )
        return path
    
    def is_small_model(self, model_name: str, auxiliary_model: str) -> bool:
        """Return whether the given model should be treated as a small-scale model."""
        basic_config = self.get_basic_config(auxiliary_model)
        small_models = basic_config.get("small_models", [])
        return model_name in small_models
    
    def get_eicl_params(self, auxiliary_model: str, model_name: str, data_type: str) -> Dict:
        """
        Get EICL-specific hyper-parameters.
        
        Priority (new format):
        1) {AUX}_auxiliary_model -> DATASET -> MODEL
        2) {AUX}_auxiliary_model -> DATASET -> default
        3) {AUX}_auxiliary_model -> default
        4) fallback to legacy keys: model_specific_params > dataset_specific_params > auxiliary_model_specific_params > EICL_params
        """
        default_params = {
            "alpha": 0.2,
            "label_score_num": 5,
            "label_num": 2,
        }
        if not self.eicl_config:
            return default_params

        params = dict(self.eicl_config.get("EICL_params", default_params))

        # ---------- New nested format ----------
        aux_key = None
        target_key = f"{auxiliary_model}_auxiliary_model".lower()
        for k in self.eicl_config.keys():
            if k.lower() == target_key:
                aux_key = k
                break

        if aux_key:
            aux_block = self.eicl_config.get(aux_key, {})
            # aux-level default
            if isinstance(aux_block, dict):
                params.update(aux_block.get("default", {}))

                # dataset section (try upper, original, lower)
                ds_block = (
                    aux_block.get(data_type.upper())
                    or aux_block.get(data_type)
                    or aux_block.get(data_type.lower())
                    or {}
                )
                if isinstance(ds_block, dict):
                    # dataset-level default
                    params.update(ds_block.get("default", {}))

                    # model-level (case-insensitive)
                    model_params = None
                    if model_name in ds_block:
                        model_params = ds_block.get(model_name, {})
                    else:
                        for mk, mv in ds_block.items():
                            # exact case-insensitive
                            if mk.lower() == model_name.lower():
                                model_params = mv
                                break
                            # fuzzy match for GPT/Claude style names
                            if mk.lower() in ["gpt", "chatgpt"] and "gpt" in model_name.lower():
                                model_params = mv
                                break
                            if mk.lower() == "claude" and "claude" in model_name.lower():
                                model_params = mv
                                break
                    if model_params:
                        params.update(model_params)

        # ---------- Legacy format fallback ----------
        else:
            # Prefer model-specific parameters
            model_params = self.eicl_config.get("model_specific_params", {}).get(model_name, {})
            if model_params:
                params.update(model_params)

            # Then use dataset-specific parameters
            dataset_params = self.eicl_config.get("dataset_specific_params", {}).get(data_type.lower(), {})
            if dataset_params:
                params.update(dataset_params)

            # Finally use auxiliary-model-specific parameters
            aux_params = self.eicl_config.get("auxiliary_model_specific_params", {}).get(auxiliary_model.upper(), {})
            if aux_params:
                params.update(aux_params)

        return {
            "alpha": params.get("alpha", 0.2),
            "label_score_num": params.get("label_score_num", 5),
            "label_num": params.get("label_num", 2),
            "generate_batch_size": params.get("generate_batch_size", None),
        }
    
    def build_config(self, 
                    auxiliary_model: str,
                    data_name: str,
                    data_type: str,
                    experiment_type: str,
                    model_name: str) -> Config:
        """Build a full ``Config`` object."""
        basic_config = self.get_basic_config(auxiliary_model)
        
        # Determine whether it is a small-scale model
        is_small = self.is_small_model(model_name, auxiliary_model)
        
        # Get EICL parameters if needed
        eicl_params = {}
        if experiment_type.upper() == "EICL":
            eicl_params = self.get_eicl_params(auxiliary_model, model_name, data_type)
        
        # Get embedding model type
        embed_model = None
        if basic_config.get("embed_model_auto", True):
            if experiment_type.upper() == "EICL":
                embed_model = basic_config.get("default_embed_model", {}).get("EICL", "emotion")
            else:
                embed_model = basic_config.get("default_embed_model", {}).get("default", "semantic")
        
        # Build DataConfig with data_batch_size (for data building / embedding)
        data_batch_size = self._resolve_overrides(
            basic_config,
            base_key="data_batch_size",
            override_key="data_batch_size_overrides",
            model_name=model_name,
            experiment_type=experiment_type,
        )

        # Split EICL params: DataConfig does not accept generate_batch_size
        data_eicl_params = dict(eicl_params)
        data_eicl_params.pop("generate_batch_size", None)

        data_config = DataConfig(
            data_root=basic_config.get("data_root", "data"),
            output_root=basic_config.get("json_output_root", "data/json_data"),
            runtime_data_path=basic_config.get("runtime_data_path", "data/json_data"),
            auxiliary_model=auxiliary_model,
            data_name=data_name,
            data_type=data_type,
            batch_size=data_batch_size,
            use_gpu=basic_config.get("use_gpu", False),
            device_id=basic_config.get("device_id", "0"),
            embed_model=embed_model,
            regenerate_embeddings=False,
            regenerate_examples=False,
            regenerate_emo_prob=False,
            **data_eicl_params
        )
        
        # Get instruction content
        instruction_content = self.get_instruction(auxiliary_model, experiment_type, data_type)
        
        # Build ExperimentConfig
        experiment_config = ExperimentConfig(
            experiment_type=experiment_type,
            example_num=basic_config.get("example_num", 5),
            instruction_path=None,
            instruction_content=instruction_content,
            instruction_base_path=None
        )
        
        # Build ModelConfig with generation_batch_size overrides (used during inference)
        generate_batch_size = self._resolve_overrides(
            basic_config,
            base_key="generation_batch_size",
            override_key="generation_batch_size_overrides",
            model_name=model_name,
            experiment_type=experiment_type,
        )
        if experiment_type.upper() == "EICL":
            if eicl_params.get("generate_batch_size") is not None:
                generate_batch_size = eicl_params["generate_batch_size"]
        
        model_config = ModelConfig(
            model_name=model_name,
            use_api=not is_small,
            api_url=None,
            api_key=None,
            api_model=None,
            model_path=None,
            model_base_path=basic_config.get("model_base_path", "./models/LLMs"),
            generate_batch_size=generate_batch_size,
            torch_dtype="float16"
        )
        
        # Build OutputConfig
        output_config = OutputConfig(
            output_base_path="./output"
        )
        
        # Create and return final Config
        config = Config(
            data=data_config,
            experiment=experiment_config,
            model=model_config,
            output=output_config
        )
        
        return config

