"""
Result summary module - write experiment results into Result_summary.txt.
"""
import os
import json
import datetime
from typing import Dict, List, Optional
from pathlib import Path


class ResultSummary:
    """Result summarizer responsible for aggregating experiment metrics."""
    
    def __init__(self, summary_file: str = "output/Result_summary.txt"):
        self.summary_file = summary_file
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    
    def format_result_entry(self, 
                          auxiliary_model: str,
                          experiment_type: str,
                          data_name: str,
                          data_type: str,
                          model_name: str,
                          accuracy: float,
                          macro_f1: float,
                          distinct_1: float = 0.0,
                          distinct_2: float = 0.0,
                          num_correct: float = 0.0,
                          num_samples: float = 0.0,
                          output_path: str = "") -> str:
        """
        Format a single result entry.

        The format is aligned with ed_result.txt for easier comparison, with
        additional fields for the number of correct predictions and total
        evaluated samples.
        """
        current_time = datetime.datetime.now()
        exp_id = f"{data_type}_{model_name}_{experiment_type}_{auxiliary_model}"
        header = (
            f"{current_time}\t{exp_id}\t[Acc]\t[Macro_f1]\t[Correct]\t[Total]\n"
        )
        values = (
            f"{accuracy:.3f}, {macro_f1:.3f}, {num_correct:.3f}, {num_samples:.3f}\n"
        )
        extra = f"Output path: {output_path}\n\n"
        return header + values + extra
    
    def add_result(self,
                  auxiliary_model: str,
                  experiment_type: str,
                  data_name: str,
                  data_type: str,
                  model_name: str,
                  accuracy: float,
                  macro_f1: float,
                  distinct_1: float = 0.0,
                  distinct_2: float = 0.0,
                  num_correct: float = 0.0,
                  num_samples: float = 0.0,
                  output_path: str = ""):
        """Append a single experiment result to the summary file."""
        entry = self.format_result_entry(
            auxiliary_model=auxiliary_model,
            experiment_type=experiment_type,
            data_name=data_name,
            data_type=data_type,
            model_name=model_name,
            accuracy=accuracy,
            macro_f1=macro_f1,
            distinct_1=distinct_1,
            distinct_2=distinct_2,
            num_correct=num_correct,
            num_samples=num_samples,
            output_path=output_path
        )
        
        with open(self.summary_file, "a", encoding="utf-8") as f:
            f.write(entry)
    
    def add_batch_results(self, results: List[Dict]):
        """Append a batch of experiment results to the summary file."""
        for result in results:
            self.add_result(**result)
    
    def create_summary_header(self, config_info: Dict):
        """Create the header section for the summary file."""
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"""
{'#'*100}
Experiment result summary
Generated at: {current_time}
Configuration:
  {json.dumps(config_info, indent=2, ensure_ascii=False)}
{'#'*100}

"""
        with open(self.summary_file, "a", encoding="utf-8") as f:
            f.write(header)

