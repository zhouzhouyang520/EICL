# Fine-Grained Emotion Recognition via In-Context Learning

This repository contains the code for the paper **"Fine-Grained Emotion Recognition via In-Context Learning"** published at CIKM 2025.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Data Download](#data-download)
- [Running Instructions](#running-instructions)
- [Performance](#performance)
- [Citation](#citation)

## Environment Setup

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended for local models)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/zhouzhouyang520/EICL.git
cd EICL
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Data Download

Download the required data from one of the following sources:

- **Baidu Netdisk**: [EICL_data](https://pan.baidu.com/s/12SERcBjWpVFJBLhP45S_Ww?pwd=i3tn) (Extraction code: `i3tn`)
- **Google Drive**: [EICL_data](https://drive.google.com/drive/folders/1nGsdwO1jXfpGiE8bfIQ0fUpFUmXJgsCv?usp=sharing)

### Data Setup

1. **Extract data**: After downloading, extract `data.zip` and place it in the EICL root directory.

2. **EmpatheticIntents** (Optional, only needed if you need to rebuild data):
   - Extract `EmpatheticIntents.zip` and place it in the EICL root directory.

3. **Build models folder** with the following structure:
   ```
   models/
   ├── LLMs/
   │   ├── Llama3.1_8b/
   │   ├── Mistral_Nemo/
   │   └── Phi3.5_mini/
   └── pre_trained_models/  (Optional, only needed if you need to rebuild data)
       ├── EmpatheticIntents/
       ├── all_mpnet_base_v2/
       └── roberta_large_goEmotions/
   ```

   - **LLMs**: Download three large language models and place them in `models/LLMs/` with the exact folder names: `Llama3.1_8b`, `Mistral_Nemo`, and `Phi3.5_mini`.
   
   - **pre_trained_models** (Optional): If you need to rebuild data, download three pre-trained models:
     - `EmpatheticIntents`
     - `all_mpnet_base_v2`
     - `roberta_large_goEmotions`

## Running Instructions

### Basic Usage

Run experiments using the main script:

```bash
python main.py \
  --auxiliary_model EI \
  --experiment_type EICL \
  --dataset ED \
  --models Phi-3.5
```

### Arguments

- `--auxiliary_model`: Auxiliary model type, choices: `EI`, `GE` (can specify multiple)
- `--experiment_type`: Experiment type, choices: `baseline`, `ICL`, `EICL`, `zero-shot` (can specify multiple)
- `--dataset`: Dataset name, choices: `ED`, `EDOS`, `GE`, `EI` (can specify multiple)
- `--models`: LLM model names, e.g., `Phi-3.5`, `Llama3.1_8b`, `Mistral-Nemo`, `ChatGPT`, `Claude`, `gpt-4o-mini` (can specify multiple)

### Examples

**Run EICL experiment with Phi-3.5 on ED dataset:**
```bash
python main.py --auxiliary_model GE --experiment_type EICL --dataset ED --models Phi-3.5
```

**Run multiple experiments:**
```bash
python main.py \
  --auxiliary_model EI GE \
  --experiment_type ICL EICL \
  --dataset ED EDOS \
  --models Phi-3.5 Llama3.1_8b
```

**Run with GPU:**
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --auxiliary_model GE \
  --experiment_type EICL \
  --dataset EDOS \
  --models Phi-3.5
```

**Using the provided shell script:**
```bash
sh eicl.sh 0  # 0 is the GPU ID
```

## Performance

**Important Note**: Due to accidental deletion of previous data and code, we reconstructed the codebase in a different environment. The current implementation differs from the original paper in the following aspects:
(1) Data Re-splitting: The datasets were re-partitioned, which may cause slight performance differences.

(2) Updated Models: We use more advanced large language models—Claude-Haiku-4.5 and GPT-4o-mini. Although the original EICL experiments used Claude-Haiku and ChatGPT-3.5-Turbo, those models are now outdated. To build more comparable and up-to-date baselines, we adopted the newer models.

Despite these differences, EICL continues to deliver excellent performance, demonstrating its robustness and effectiveness.

The following tables present the performance results (Accuracy and Macro F1) on different datasets using different auxiliary models. All results are reported as percentages.

### Results with EI Auxiliary Model

<table>
<thead>
<tr>
<th rowspan="2">Dataset</th>
<th rowspan="2">Metric</th>
<th colspan="3">Phi-3.5-mini</th>
<th colspan="3">Mistral-Nemo</th>
<th colspan="3">Llama3.1-8b</th>
<th colspan="3">Claude-Haiku</th>
<th colspan="3">ChatGPT (GPT)</th>
</tr>
<tr>
<th>Zero-shot</th>
<th>ICL</th>
<th>EICL</th>
<th>Zero-shot</th>
<th>ICL</th>
<th>EICL</th>
<th>Zero-shot</th>
<th>ICL</th>
<th>EICL</th>
<th>Zero-shot</th>
<th>ICL</th>
<th>EICL</th>
<th>Zero-shot</th>
<th>ICL</th>
<th>EICL</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2"><strong>EDOS</strong></td>
<td>Acc</td>
<td>33.29</td>
<td>48.28</td>
<td><strong>56.38</strong></td>
<td>28.98</td>
<td>44.37</td>
<td>56.15</td>
<td>23.09</td>
<td>30.79</td>
<td>41.87</td>
<td>36.03</td>
<td>49.27</td>
<td>57.14</td>
<td>32.89</td>
<td>44.2</td>
<td>54.75</td>
</tr>
<tr>
<td>F1</td>
<td>34.39</td>
<td>47.84</td>
<td><strong>56.37</strong></td>
<td>26.69</td>
<td>44.67</td>
<td>56.24</td>
<td>21.31</td>
<td>34.08</td>
<td>45.33</td>
<td>34.21</td>
<td>48.29</td>
<td>57.04</td>
<td>30.19</td>
<td>44.79</td>
<td>54.27</td>
</tr>
<tr>
<td rowspan="2"><strong>ED</strong></td>
<td>Acc</td>
<td>35.69</td>
<td>44.92</td>
<td><strong>48.08</strong></td>
<td>37.27</td>
<td>38.44</td>
<td>45.08</td>
<td>32.42</td>
<td>34.02</td>
<td>38.97</td>
<td>46.03</td>
<td>50.2</td>
<td>51.36</td>
<td>39.3</td>
<td>47.04</td>
<td>50.31</td>
</tr>
<tr>
<td>F1</td>
<td>34.04</td>
<td>44.3</td>
<td><strong>47.94</strong></td>
<td>34.73</td>
<td>38.48</td>
<td>44.94</td>
<td>25.71</td>
<td>36.27</td>
<td>42.2</td>
<td>43.76</td>
<td>49.32</td>
<td>51.09</td>
<td>35.24</td>
<td>45.73</td>
<td>49.72</td>
</tr>
<tr>
<td rowspan="2"><strong>GE</strong></td>
<td>Acc</td>
<td>27.51</td>
<td>40.38</td>
<td>35.85</td>
<td>30.62</td>
<td>38.05</td>
<td>30.83</td>
<td>23.94</td>
<td>22.98</td>
<td>27.57</td>
<td>34.07</td>
<td><strong>44.65</strong></td>
<td>44.16</td>
<td>28.73</td>
<td>45.44</td>
<td><strong>45.93</strong></td>
</tr>
<tr>
<td>F1</td>
<td>27.84</td>
<td>33.34</td>
<td>32.79</td>
<td>26.46</td>
<td>31.45</td>
<td>28.90</td>
<td>22.15</td>
<td>20.46</td>
<td>27.43</td>
<td>31.51</td>
<td><strong>37.36</strong></td>
<td>36.36</td>
<td>26.08</td>
<td>36.67</td>
<td><strong>36.96</strong></td>
</tr>
</tbody>
</table>

### Results with GE Auxiliary Model

<table>
<thead>
<tr>
<th rowspan="2">Dataset</th>
<th rowspan="2">Metric</th>
<th colspan="3">Phi-3.5-mini</th>
<th colspan="3">Mistral-Nemo</th>
<th colspan="3">Llama3.1-8b</th>
<th colspan="3">Claude-Haiku</th>
<th colspan="3">ChatGPT-Turbo</th>
</tr>
<tr>
<th>Zero-shot</th>
<th>ICL</th>
<th>EICL</th>
<th>Zero-shot</th>
<th>ICL</th>
<th>EICL</th>
<th>Zero-shot</th>
<th>ICL</th>
<th>EICL</th>
<th>Zero-shot</th>
<th>ICL</th>
<th>EICL</th>
<th>Zero-shot</th>
<th>ICL</th>
<th>EICL</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2"><strong>EDOS</strong></td>
<td>Acc</td>
<td>52.46</td>
<td>60.27</td>
<td>58.13</td>
<td>47.54</td>
<td>52.08</td>
<td>52.96</td>
<td>38.34</td>
<td>35.68</td>
<td>50.94</td>
<td>51.07</td>
<td>63.05</td>
<td><strong>69.48</strong></td>
<td>52.84</td>
<td>61.41</td>
<td>65.45</td>
</tr>
<tr>
<td>F1</td>
<td>50.41</td>
<td>59.85</td>
<td>59.79</td>
<td>45.07</td>
<td>51.2</td>
<td>57.24</td>
<td>38.86</td>
<td>39.23</td>
<td>55.54</td>
<td>49.36</td>
<td>62.44</td>
<td><strong>69.11</strong></td>
<td>51.29</td>
<td>60.89</td>
<td>65.8</td>
</tr>
<tr>
<td rowspan="2"><strong>ED</strong></td>
<td>Acc</td>
<td>45.1</td>
<td>47.13</td>
<td>47.4</td>
<td>48.76</td>
<td>45.07</td>
<td>51.93</td>
<td>44.87</td>
<td>30.79</td>
<td>45.7</td>
<td>55.06</td>
<td>57.26</td>
<td><strong>60.82</strong></td>
<td>52.66</td>
<td>57.66</td>
<td>59.25</td>
</tr>
<tr>
<td>F1</td>
<td>46.1</td>
<td>48.33</td>
<td>50.26</td>
<td>49.36</td>
<td>46.69</td>
<td>53.46</td>
<td>45.94</td>
<td>35.65</td>
<td>51.05</td>
<td>56.26</td>
<td>57.94</td>
<td><strong>61.57</strong></td>
<td>54.07</td>
<td>58.79</td>
<td>60.54</td>
</tr>
<tr>
<td rowspan="2"><strong>EI</strong></td>
<td>Acc</td>
<td>47.47</td>
<td>63.86</td>
<td>60.56</td>
<td>53.33</td>
<td>61.55</td>
<td>65.30</td>
<td>42.05</td>
<td>41.93</td>
<td>51.03</td>
<td>58.01</td>
<td>68.59</td>
<td><strong>70.09</strong></td>
<td>55.58</td>
<td>68.34</td>
<td>68.34</td>
</tr>
<tr>
<td>F1</td>
<td>46.62</td>
<td>64.34</td>
<td>61.89</td>
<td>52.23</td>
<td>62.03</td>
<td>66.39</td>
<td>43.30</td>
<td>48.32</td>
<td>57.48</td>
<td>57.38</td>
<td>69.40</td>
<td><strong>70.70</strong></td>
<td>54.64</td>
<td>68.44</td>
<td>68.83</td>
</tr>
</tbody>
</table>

For more detailed results and analysis, please refer to: [Performance Results](https://drive.google.com/drive/folders/1nGsdwO1jXfpGiE8bfIQ0fUpFUmXJgsCv?usp=sharing)

## Citation

If you find this project helpful, please cite our paper:

```bibtex
@inproceedings{Ren_2025,
  series={CIKM' 25},
  title={Fine-Grained Emotion Recognition via In-Context Learning},
  url={http://dx.doi.org/10.1145/3746252.3761319},
  DOI={10.1145/3746252.3761319},
  booktitle={Proceedings of the 34th ACM International Conference on Information and Knowledge Management},
  publisher={ACM},
  author={Ren, Zhaochun and Yang, Zhou and Ye, Chenglong and Sun, Haizhou and Chen, Chao and Zhu, Xiaofei and Liao, Xiangwen},
  year={2025},
  month=nov,
  pages={2503--2513},
  collection={CIKM' 25}
}
```



=======
# EICL
This is the code for the paper Fine-Grained Emotion Recognition via In-Context Learning, published at CIKM 2025.
>>>>>>> 224697e2136eb23dce4e47c6bec8d69fbf4dcf66



