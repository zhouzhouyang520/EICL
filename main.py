#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main entry file - unified orchestration and control logic.
Use ``main.py`` as the entry point to run experiments.
Supports batch execution and result summarization.
Based on the basic configuration files (``EI_basic_config.json``, ``GE_basic_config.json``, ``EICL_config.json``, ``run.json).
"""
import os
import sys
import argparse
from typing import List, Dict, Optional

# Add path for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.config_loader import ConfigLoader
from modules.data_builder import DataBuilder
from modules.runner import ModelRunner
from modules.evaluator import Evaluator
from modules.result_summary import ResultSummary
from modules.config import Config


def print_config(config: Config) -> None:
    """Print the effective configuration that will be used to run the model."""
    data_cfg = config.data
    exp_cfg = config.experiment
    model_cfg = config.model
    out_cfg = config.output

    print("\n[Effective Configuration]")
    print("Data:")
    print(f"  data_root: {data_cfg.data_root}")
    print(f"  output_root: {data_cfg.output_root}")
    print(f"  runtime_data_path: {data_cfg.runtime_data_path}")
    print(f"  auxiliary_model: {data_cfg.auxiliary_model}")
    print(f"  data_name: {data_cfg.data_name}")
    print(f"  data_type: {data_cfg.data_type}")
    print(f"  batch_size: {data_cfg.batch_size}")
    print(f"  use_gpu: {data_cfg.use_gpu}")
    print(f"  device_id: {data_cfg.device_id}")
    print(f"  embed_model: {data_cfg.embed_model}")
    print(f"  regenerate_embeddings: {data_cfg.regenerate_embeddings}")
    print(f"  regenerate_examples: {data_cfg.regenerate_examples}")
    print(f"  regenerate_emo_prob: {data_cfg.regenerate_emo_prob}")
    print(f"  alpha: {data_cfg.alpha}")
    print(f"  label_score_num: {data_cfg.label_score_num}")
    print(f"  label_num: {data_cfg.label_num}")

    print("Experiment:")
    print(f"  experiment_type: {exp_cfg.experiment_type}")
    print(f"  example_num: {exp_cfg.example_num}")
    print(f"  instruction_path: {exp_cfg.instruction_path}")
    print(f"  instruction_content set: {exp_cfg.instruction_content is not None}")

    print("Model:")
    print(f"  model_name: {model_cfg.model_name}")
    print(f"  use_api: {model_cfg.use_api}")
    print(f"  api_url: {model_cfg.api_url}")
    print(f"  api_key: {model_cfg.api_key}")
    print(f"  api_model: {model_cfg.api_model}")
    print(f"  model_path: {model_cfg.model_path}")
    print(f"  model_base_path: {model_cfg.model_base_path}")
    print(f"  generate_batch_size: {model_cfg.generate_batch_size}")
    print(f"  torch_dtype: {model_cfg.torch_dtype}")
    print(f"  use_fast_tokenizer: {model_cfg.use_fast_tokenizer}")

    print("Output:")
    print(f"  output_base_path: {out_cfg.output_base_path}")


def get_data_type_mapping(data_name: str) -> str:
    """Map dataset name to its data type string."""
    mapping = {
        "ED": "ed",
        "EDOS": "edos",
        "GE": "ge",
        "EI": "ei"
    }
    return mapping.get(data_name.upper(), data_name.lower())


def run_single_experiment(
    dataset: str,
    model_name: str,
    experiment_type: str,
    auxiliary_model: str,
    config_loader: ConfigLoader,
    skip_data_check: bool = False,
    skip_evaluation: bool = False
) -> Optional[Dict]:
    """
    Run a single experiment.

    Returns:
        A dictionary containing evaluation results. Returns ``None`` if the run fails.
    """
    print("\n" + "=" * 100)
    print(f"Start experiment: dataset={dataset}, model={model_name}, "
          f"exp_type={experiment_type}, aux_model={auxiliary_model}")
    print("=" * 100)
    
    # Get data type
    data_type = get_data_type_mapping(dataset)
    
    # Skip incompatible combinations
    if dataset.upper() == auxiliary_model.upper():
        print(f"Skip incompatible combination: dataset={dataset}, aux_model={auxiliary_model}")
        return None
    
    # 1. Build config using ConfigLoader
    try:
        config = config_loader.build_config(
            auxiliary_model=auxiliary_model,
            data_name=dataset,
            data_type=data_type,
            experiment_type=experiment_type,
            model_name=model_name
        )
    except Exception as e:
        print(f"Failed to build config: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    print("Configuration built successfully")
    print(f"  Model: {config.model.model_name}")
    print(f"  Experiment type: {config.experiment.experiment_type}")
    print(f"  Auxiliary model: {config.data.auxiliary_model}")
    print(f"  Dataset: {config.data.data_name} ({config.data.data_type})")
    print(f"  Run mode: {'API' if config.model.use_api else 'Local'}")
    # Print the effective config values (after any in-code adjustments)
    print_config(config)
    
    # 2. Build data
    if not skip_data_check:
        print("\n[Step 1] Building data...")
        try:
            data_builder = DataBuilder(config)
            if not data_builder.build_data():
                print("Data building failed, skip this experiment")
                return None
            print("Data building finished")
        except Exception as e:
            print(f"Error occurred while building data: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        print("\n[Step 1] Skip data checking")
    
    # 3. Run model
    print("\n[Step 2] Running model...")
    try:
        runner = ModelRunner(config)
        output_path, eval_result, _ = runner.run()
        
        if not output_path:
            print("Model run failed, skip this experiment")
            return None
        
        print(f"Model run finished, output path: {output_path}")
    except Exception as e:
        print(f"Error occurred while running model: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 4. Evaluate results
    exp_id = config.get_experiment_id()
    
    if not skip_evaluation:
        print("\n[Step 3] Evaluating results...")
        try:
            # Use evaluation results returned by model run if available
            if eval_result and isinstance(eval_result, dict) and 'accuracy' in eval_result:
                print("Using evaluation results returned by model run")
                result_dict = {
                    'auxiliary_model': auxiliary_model,
                    'experiment_type': experiment_type,
                    'data_name': dataset,
                    'data_type': config.data.data_type,
                    'model_name': model_name,
                    'accuracy': eval_result.get('accuracy', 0.0),
                    'macro_f1': eval_result.get('macro_f1', 0.0),
                    'distinct_1': eval_result.get('distinct_1', 0.0),
                    'distinct_2': eval_result.get('distinct_2', 0.0),
                    'num_correct': eval_result.get('num_correct', 0.0),
                    'num_samples': eval_result.get('num_samples', 0.0),
                    'output_path': output_path
                }
            else:
                # Use ``Evaluator`` to re-evaluate
                evaluator = Evaluator(
                    auxiliary_model=config.data.auxiliary_model,
                    data_type=config.data.data_type
                )
                eval_result_dict = evaluator.evaluate(output_path, exp_id)
                
                # Build result dictionary
                result_dict = {
                    'auxiliary_model': auxiliary_model,
                    'experiment_type': experiment_type,
                    'data_name': dataset,
                    'data_type': config.data.data_type,
                    'model_name': model_name,
                    'accuracy': eval_result_dict.get('accuracy', 0.0),
                    'macro_f1': eval_result_dict.get('macro_f1', 0.0),
                    'distinct_1': eval_result_dict.get('distinct_1', 0.0),
                    'distinct_2': eval_result_dict.get('distinct_2', 0.0),
                    'num_correct': eval_result_dict.get('num_correct', 0.0),
                    'num_samples': eval_result_dict.get('num_samples', 0.0),
                    'output_path': output_path
                }
            
            print("Evaluation finished")
            print(f"  Accuracy: {result_dict['accuracy']:.2f}%")
            print(f"  Macro F1: {result_dict['macro_f1']:.2f}%")
            print(f"  Distinct-1: {result_dict['distinct_1']:.4f}")
            print(f"  Distinct-2: {result_dict['distinct_2']:.4f}")
            
        except Exception as e:
            print(f"Error occurred during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        print("\n[Step 3] Skip evaluation")
        result_dict = {
            'auxiliary_model': auxiliary_model,
            'experiment_type': experiment_type,
            'data_name': dataset,
            'data_type': config.data.data_type,
            'model_name': model_name,
            'accuracy': 0.0,
            'macro_f1': 0.0,
            'distinct_1': 0.0,
            'distinct_2': 0.0,
            'num_correct': 0.0,
            'num_samples': 0.0,
            'output_path': output_path
        }
    
    print("=" * 100)
    print(f"Experiment finished: dataset={dataset}, model={model_name}, "
          f"exp_type={experiment_type}, aux_model={auxiliary_model}")
    print("=" * 100 + "\n")
    
    return result_dict


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Main entry of EICL experiments - supports batch runs and result summarization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core arguments - simplified
    parser.add_argument("--auxiliary_model", type=str, nargs="+", choices=["EI", "GE"],
                       default=["EI"], help="List of auxiliary models (multiple values allowed)")
    parser.add_argument("--experiment_type", type=str, nargs="+",
                       choices=["baseline", "ICL", "EICL", "zero-shot"],
                       default=["ICL"], help="List of experiment types (multiple values allowed)")
    parser.add_argument("--dataset", type=str, nargs="+", choices=["ED", "EDOS", "GE", "EI"],
                       default=["ED"], help="List of datasets (multiple values allowed)")
    parser.add_argument("--models", type=str, nargs="+",
                       default=["Phi-3.5"], help="List of LLM models (multiple values allowed, "
                                                 "e.g.: Phi-3.5 Llama3.1_8b Mistral-Nemo ChatGPT Claude)")
    
    # Other arguments
    parser.add_argument("--skip_data_check", action="store_true",
                       help="Skip data checking")
    parser.add_argument("--skip_evaluation", action="store_true",
                       help="Skip evaluation")
    parser.add_argument("--config_dir", type=str, default="configs",
                       help="Directory containing configuration files")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    #import ipdb; ipdb.set_trace()
    print("\n" + "=" * 100)
    print("Batch running EICL experiments".center(100))
    print("=" * 100)
    print(f"Auxiliary models: {args.auxiliary_model}")
    print(f"Experiment types: {args.experiment_type}")
    print(f"Datasets: {args.dataset}")
    print(f"Model list: {args.models}")
    print("=" * 100 + "\n")
    
    # Initialize configuration loader
    try:
        config_loader = ConfigLoader(config_dir=args.config_dir)
        #import ipdb; ipdb.set_trace()
        print(f"Configuration loader initialized successfully, config directory: {args.config_dir}")
    except Exception as e:
        print(f"Failed to initialize configuration loader: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create result summarizer
    result_summary = ResultSummary()
    
    # Record configuration info
    config_info = {
        'auxiliary_models': args.auxiliary_model,
        'experiment_types': args.experiment_type,
        'datasets': args.dataset,
        'models': args.models,
        'config_dir': args.config_dir
    }
    result_summary.create_summary_header(config_info)
    
    # Run experiments in batch
    all_results = []
    total_experiments = 0
    current_experiment = 0
    
    # Calculate total number of experiments (excluding incompatible combinations)
    #import ipdb; ipdb.set_trace()
    for auxiliary_model in args.auxiliary_model:
        for experiment_type in args.experiment_type:
            for dataset in args.dataset:
                if dataset.upper() != auxiliary_model.upper():
                    total_experiments += len(args.models)
    
    for auxiliary_model in args.auxiliary_model:
        for experiment_type in args.experiment_type:
            for dataset in args.dataset:
                # Skip incompatible combinations (e.g. GE dataset does not need GE as auxiliary model)
                if dataset.upper() == auxiliary_model.upper():
                    continue
                
                for model_name in args.models:
                    current_experiment += 1
                    print(f"\nProgress: {current_experiment}/{total_experiments}")
                    
                    # Run a single experiment
                    result = run_single_experiment(
                        dataset=dataset,
                        model_name=model_name,
                        experiment_type=experiment_type,
                        auxiliary_model=auxiliary_model,
                        config_loader=config_loader,
                        skip_data_check=args.skip_data_check,
                        skip_evaluation=args.skip_evaluation
                    )
                    
                    # Record results
                    if result:
                        all_results.append(result)
                        result_summary.add_result(**result)
    
    # Output summary information
    print("\n" + "=" * 100)
    print("Batch experiments finished".center(100))
    print("=" * 100)
    print(f"Total experiments: {total_experiments}")
    print(f"Succeeded: {len(all_results)}")
    print(f"Failed: {total_experiments - len(all_results)}")
    print(f"Result summary file: {result_summary.summary_file}")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
