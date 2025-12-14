"""
Configuration module.
Contains data, experiment, model, and output configuration structures.
"""
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path


@dataclass
class DataConfig:
    """Data configuration - controls parameters passed to build_data.py."""
    data_root: str = "data"
    output_root: str = "data/json_data"  # Output path for build_data.py
    runtime_data_path: str = "data/json_data"  # Runtime data path (may differ from output_root)
    auxiliary_model: str = "EI"  # EI or GE
    data_name: str = "ED"  # ED, EDOS, GE
    data_type: str = ""  # Defaults to lower-cased data_name
    batch_size: int = 16
    use_gpu: bool = False
    device_id: str = "0"
    embed_model: Optional[str] = None  # "semantic" or "emotion"; None means auto-select
    regenerate_embeddings: bool = False
    regenerate_examples: bool = False
    regenerate_emo_prob: bool = False
    
    # EICL-specific parameters - passed through to build_data.py helpers
    alpha: float = 0.2  # Alpha parameter for construct_emotions
    label_score_num: int = 5  # label_score_num parameter for format_example
    label_num: int = 2  # label_num parameter for format_context


@dataclass
class ExperimentConfig:
    """Experiment configuration - parameters for baseline / ICL / EICL runs."""
    experiment_type: str = "ICL"  # baseline, ICL, EICL
    example_num: int = 5  # Number of in-context examples
    
    # instruct.txt path selection - determined by dataset and experiment type
    # Can be set explicitly or inferred from default rules
    instruction_path: Optional[str] = None  # None means use default path rule
    
    # Instruction content (if set, takes precedence over file-based instruction)
    instruction_content: Optional[str] = None
    
    # Default instruction path rule
    instruction_base_path: str = "eval_data/ed_json_data"
    
    def get_instruction_path(self, data_type: str, auxiliary_model: str) -> str:
        """Get instruction file path based on dataset and experiment type."""
        if self.instruction_path:
            return self.instruction_path
        
        # Default path rule
        if self.experiment_type.upper() == "BASELINE":
            return f"{self.instruction_base_path}/baseline_instruction.txt"
        elif self.experiment_type.upper() == "ICL":
            return f"{self.instruction_base_path}/icl_instruction.txt"
        elif self.experiment_type.upper() == "EICL":
            return f"{self.instruction_base_path}/eicl_instruction.txt"
        else:
            return f"{self.instruction_base_path}/instruction.txt"
    
    def get_instruction(self, data_type: str, auxiliary_model: str) -> Optional[str]:
        """Return instruction content, preferring the in-memory ``instruction_content``."""
        if self.instruction_content:
            return self.instruction_content
        return None


@dataclass
class ModelConfig:
    """Model configuration - controls API or local model execution."""
    # Model type
    model_name: str = "Phi-3.5"  # Phi-3.5, Llama3.1_8b, Mistral-Nemo, ChatGPT, Claude, etc.
    
    # Execution mode
    use_api: bool = False  # True: run via API; False: run locally
    
    # API configuration
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    api_model: Optional[str] = None  # API model name; if None, inferred from model_name
    
    # Local model configuration
    model_path: Optional[str] = None  # Local model path; None means use default path
    model_base_path: str = "./models/LLMs"  # Base path for local models
    
    # Mapping from model name to actual directory name
    MODEL_NAME_TO_DIR = {
        "Phi-3.5": "Phi3.5_mini",
        "Llama3.1_8b": "Llama3.1_8b",
        "Mistral-Nemo": "Mistral_Nemo",
    }
    
    generate_batch_size: int = 2  # Generation batch size
    # Other local model parameters
    torch_dtype: str = "float16"  # float16 or float32
    use_fast_tokenizer: bool = True
    
    def get_model_path(self) -> str:
        """Get the resolved path of the model."""
        if self.model_path:
            return self.model_path
        
        # Get actual directory name
        actual_dir_name = self.MODEL_NAME_TO_DIR.get(self.model_name, self.model_name)
        return f"{self.model_base_path}/{actual_dir_name}"
    
    def is_small_model(self) -> bool:
        """Return True if the model is considered a small local model."""
        small_models = ["Phi-3.5", "Llama3.1_8b", "Mistral-Nemo"]
        return self.model_name in small_models
    
    def is_large_model(self) -> bool:
        """Return True if the model is considered a large API model."""
        large_model_keywords = ["ChatGPT", "Claude", "gpt", "claude"]
        return any(keyword.lower() in self.model_name.lower() for keyword in large_model_keywords)


@dataclass
class OutputConfig:
    """Output configuration - controls where results are saved."""
    output_base_path: str = "./output"
    
    def get_output_dir(self, experiment_type: str, auxiliary_model: str) -> str:
        """Get output directory for a given experiment type and auxiliary model."""
        if experiment_type.upper() == "BASELINE":
            exp_path = "baseline"
        elif experiment_type.upper() == "ZERO-SHOT":
            exp_path = ""
        else:
            exp_path = experiment_type.upper()
        
        return str(Path(self.output_base_path) / exp_path / auxiliary_model)
    
    def get_output_filename(self, data_type: str, model_name: str, 
                           experiment_type: str, auxiliary_model: str) -> str:
        """Generate the output file name for predictions."""
        exp_id = f"{data_type}_{model_name}_{experiment_type}_{auxiliary_model}"
        return f"{exp_id}_generated_predictions.jsonl"


@dataclass
class Config:
    """Full configuration class that aggregates all configuration sections."""
    data: DataConfig
    experiment: ExperimentConfig
    model: ModelConfig
    output: OutputConfig
    
    def __post_init__(self):
        """Post-init hook for automatic field adjustments."""
        # Automatically set use_api
        if self.model.is_large_model():
            self.model.use_api = True
        else:
            self.model.use_api = False
        
        # Set default data_type
        if not self.data.data_type:
            self.data.data_type = self.data.data_name.lower()
        
        # Set default embed_model
        if not self.data.embed_model:
            if self.experiment.experiment_type.upper() == "EICL":
                self.data.embed_model = "emotion"
            else:
                self.data.embed_model = "semantic"
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """Create a Config instance from a plain dictionary."""
        data_config = DataConfig(**config_dict.get('data', {}))
        exp_config = ExperimentConfig(**config_dict.get('experiment', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        output_config = OutputConfig(**config_dict.get('output', {}))
        
        config = cls(
            data=data_config,
            experiment=exp_config,
            model=model_config,
            output=output_config
        )
        return config
    
    def get_experiment_id(self) -> str:
        """Get a unique experiment identifier string."""
        return f"{self.data.data_type}_{self.model.model_name}_{self.experiment.experiment_type}_{self.data.auxiliary_model}"
    
    def to_dict(self) -> dict:
        """Convert the configuration into a serializable dictionary."""
        return {
            'data': self.data.__dict__,
            'experiment': self.experiment.__dict__,
            'model': self.model.__dict__,
            'output': self.output.__dict__
        }

