"""
Runner module - responsible for invoking models via API or local execution.
"""
import os
import json
import jsonlines
import sys
from pathlib import Path
from typing import Optional, Tuple, List

# Add path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.config import Config

# Import small-scale model logic
from modules.local_model_runner import run_local_model

# Import remote model runner (for ChatGPT and Claude)
from modules.remote_model_runner import run_remote_model


class ModelRunner:
    """Model runner responsible for running models and saving results."""
    
    def __init__(self, config: Config):
        self.config = config
        
    def get_data_path(self) -> str:
        """Get the input data path."""
        # Use the runtime data path (may differ from the build-time output path)
        data_path_prefix = self.config.data.runtime_data_path
        
        exp_type = self.config.experiment.experiment_type.upper()
        
        if exp_type == "ZERO-SHOT":
            exp_type_path = ""
        elif exp_type == "BASELINE":
            exp_type_path = "baseline"
        else:
            exp_type_path = exp_type
        
        data_base_path = os.path.join(data_path_prefix, exp_type_path, 
                                     self.config.data.auxiliary_model)
        
        # File naming rules:
        # - BASELINE / ICL (and ZERO-SHOT): {data_type}_{experiment_type}_{aux}_data.json
        # - EICL: {data_type}_{model_name}_{experiment_type}_{aux}_data.json
        if exp_type == "EICL":
            test_name = f"{self.config.data.data_type}_{self.config.model.model_name}_{self.config.experiment.experiment_type}_{self.config.data.auxiliary_model}_data.json"
        else:
            test_name = f"{self.config.data.data_type}_{self.config.experiment.experiment_type}_{self.config.data.auxiliary_model}_data.json"
        return os.path.join(data_base_path, test_name)
    
    def get_output_path(self) -> str:
        """Get the output file path."""
        output_dir = self.config.output.get_output_dir(
            self.config.experiment.experiment_type,
            self.config.data.auxiliary_model
        )
        os.makedirs(output_dir, exist_ok=True)
        
        filename = self.config.output.get_output_filename(
            self.config.data.data_type,
            self.config.model.model_name,
            self.config.experiment.experiment_type,
            self.config.data.auxiliary_model
        )
        return os.path.join(output_dir, filename)
    
    def run_small_model(self) -> Tuple[Optional[str], Optional[dict]]:
        """
        Run a small-scale model (local execution).

        Returns:
            (output_path, eval_result) or (None, None) on failure.
        """
        print(f"Using local small-scale model: {self.config.model.model_name}")
        
        try:
            output_path, eval_result = run_local_model(self.config)
            return output_path, eval_result
        except Exception as e:
            print(f"Small-scale model run failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def run_large_model(self) -> Tuple[Optional[str], Optional[dict]]:
        """
        Run a large-scale model via API (ChatGPT or Claude).

        Returns:
            (output_path, eval_result) on success, or (None, None) on failure.
        """
        print(f"Using large-scale API model: {self.config.model.model_name}")
        
        try:
            output_path, eval_result = run_remote_model(self.config)
            return output_path, eval_result
        except Exception as e:
            print(f"Remote model run failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def run(self) -> Tuple[Optional[str], Optional[dict], Optional[dict]]:
        """
        Run the model according to configuration.

        Returns:
            (output_path, eval_result, None) or (output_path, None, None).
        """
        if self.config.model.is_small_model() or not self.config.model.use_api:
            # Small-scale model: local invocation
            output_path, eval_result = self.run_small_model()
            return output_path, eval_result, None
        elif self.config.model.is_large_model() or self.config.model.use_api:
            # Large-scale model: API invocation (ChatGPT or Claude)
            output_path, eval_result = self.run_large_model()
            return output_path, eval_result, None
        else:
            print(f"Unknown model type: {self.config.model.model_name}, cannot run.")
            return None, None, None

