"""
Data building module - directly call ``build_data.py`` helpers to construct data.
"""
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.config import Config

# Import functions and classes from build_data.py
from data_processor.build_data import BuildOptions, run as build_data_run


class DataBuilder:
    """Data builder responsible for calling build_data.py to construct data."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def get_data_path(self) -> str:
        """Get the expected data path."""
        # Use output_root from config
        data_path_prefix = self.config.data.output_root
        
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

    def get_default_data_path(self) -> str:
        """Default path produced by build_data.py (legacy naming)."""
        data_path_prefix = self.config.data.output_root
        exp_type = self.config.experiment.experiment_type.upper()
        if exp_type == "ZERO-SHOT":
            exp_type_path = ""
        elif exp_type == "BASELINE":
            exp_type_path = "baseline"
        else:
            exp_type_path = exp_type
        data_base_path = os.path.join(data_path_prefix, exp_type_path, self.config.data.auxiliary_model)
        return os.path.join(data_base_path, f"{self.config.data.data_type}_tst.json")
    
    def check_data_exists(self) -> bool:
        """Check whether the expected data file already exists."""
        data_path = self.get_data_path()
        return os.path.exists(data_path)
    
    def _config_to_build_options(self) -> BuildOptions:
        """Convert a Config instance into a BuildOptions instance."""
        return BuildOptions(
            data_root=self.config.data.data_root,
            output_root=self.config.data.output_root,
            auxiliary_model=self.config.data.auxiliary_model,
            data_name=self.config.data.data_name,
            data_type=self.config.data.data_type,
            experiment_type=self.config.experiment.experiment_type,
            example_num=self.config.experiment.example_num,
            batch_size=self.config.data.batch_size,
            use_gpu=self.config.data.use_gpu,
            device_id=self.config.data.device_id,
            embed_model=self.config.data.embed_model,
            regenerate_embeddings=self.config.data.regenerate_embeddings,
            regenerate_examples=self.config.data.regenerate_examples,
            regenerate_emo_prob=self.config.data.regenerate_emo_prob,
            alpha=self.config.data.alpha,
            label_score_num=self.config.data.label_score_num,
            label_num=self.config.data.label_num,
        )
    
    def build_data(self) -> bool:
        """
        Build data by directly calling the run function in build_data.py.

        Returns:
            True if successful, False otherwise.
        """
        # Check if data already exists
        if self.check_data_exists():
            print(f"Data file already exists: {self.get_data_path()}")
            return True
        
        print("Data file does not exist, starting data build...")
        
        try:
            # Convert Config to BuildOptions
            build_opts = self._config_to_build_options()
            
            # Directly call the run function in build_data.py
            print("Starting data build...")
            print(f"  Data root: {build_opts.data_root}")
            print(f"  Output root: {build_opts.output_root}")
            print(f"  Auxiliary model: {build_opts.auxiliary_model}")
            print(f"  Dataset: {build_opts.data_name}")
            print(f"  Data type: {build_opts.data_type}")
            print(f"  Experiment type: {build_opts.experiment_type}")
            print(f"  Number of examples: {build_opts.example_num}")
            if build_opts.experiment_type == "EICL":
                print(f"  EICL params - alpha: {build_opts.alpha}, "
                      f"label_score_num: {build_opts.label_score_num}, "
                      f"label_num: {build_opts.label_num}")
            
            build_data_run(build_opts)
            
            # Ensure output uses new naming; rename from legacy if needed
            target_path = self.get_data_path()
            legacy_path = self.get_default_data_path()
            if (not os.path.exists(target_path)) and os.path.exists(legacy_path):
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                os.replace(legacy_path, target_path)
            
            # Check again whether the data file was generated
            if self.check_data_exists():
                print(f"Data file built successfully: {self.get_data_path()}")
                return True
            else:
                print(f"Warning: data file still not generated: {self.get_data_path()}")
                return False
                
        except Exception as e:
            print(f"Data building failed: {e}")
            import traceback
            traceback.print_exc()
            return False
