# VRU-Accident: A Vision-Language Benchmark for Video Question Answering and Dense Captioning for Accident Scene Understanding

## 🛑 Overview

**VRU-Accident** is a large-scale vision-language benchmark designed to evaluate multimodal large language models (MLLMs) in traffic accident scenarios involving **Vulnerable Road Users (VRUs)**, such as pedestrians and cyclists.

We provide:
- **1,000 real-world dashcam crash videos**
- **6,000 multiple-choice VQA pairs** across six safety-critical categories
- **1,000 dense crash scene descriptions**

This benchmark uniquely focuses on understanding the **causal, contextual, and preventive** aspects of crashes involving VRUs.

---

## 📦 Benchmark Summary

| Task              | Samples | Candidate Options | Unique Answers |
|-------------------|---------|-------------------|----------------|
| VQA (6 categories)| 6,000   | 24,000            | ~3,400         |
| Dense Captioning  | 1,000   | -                 | 1,000          |

---

## 🧠 Tasks

### 📌 Video Question Answering (VQA)
Each video includes 6 question-answer pairs with 1 correct answer and 3 contextually relevant counterfactuals.

**VQA Categories:**
- Weather & Light 
-  Traffic Environment  
- Road Configuration  
- Accident Type  
- Accident Cause  
- Accident Prevention Measure


### 📝 Dense Captioning
Each video is paired with a detailed, high-quality description capturing:
- Pedestrian/vehicle appearance
- Environmental context
- Weather condition
- Pedestrian/vehicle dynamics
- Spatial relationship between road users
- Detailed collision description

---

## 🚀 Getting Started

### 🧩 Environment

- Python 3.10
- CUDA 12.4

> Tested on Ubuntu 20.04 with NVIDIA TITAN RTX

---

### ⚙️ Installation

```bash
# Create environment
conda create -n VRU-Accident python=3.10 -y
conda activate VRU-Accident
```
```bash
# Clone repository
git clone https://github.com/Kimyounggun99/VRU-Accident.git
cd VRU-Accident
```

```bash
# Install dependencies
pip install -r requirements.txt
```

---


### 🧠 MLLM Weights Setup

You must manually download the following model weights: [Mobile-VideoGPT(1.5B)](https://huggingface.co/Amshaker/Mobile-VideoGPT-1.5B), [Video-XL-Pro(3B)](https://huggingface.co/MINT-SJTU/Video-XL-Pro-3B), and [Video-XL-2(7B)](https://huggingface.co/BAAI/Video-XL-2).

Place them under the following directory structure:

```bash
./VRU-Accident/Models/
├── Video_XL_Pro_3B/
├── Video-XL-2/
└── Mobile_VideoGPT/
    └── Mobile_VideoGPT_15B/
```

All other model weights will be downloaded automatically during evaluation.


---


### 📁 Dataset Structure

Visit our [Hugging Face page](https://huggingface.co/datasets/VRUAccidentAnonymous/VRU-Accident) to download all 1,000 accident videos and corresponding annotations. Organize the files as follows:

```bash
./VRU-Accident/
├── MetaData/
│   ├── CAP_DATA/
│   │      ├── CAP_DATA_VQA_annotation.json
│   │      └── CAP_DATA_Dense_Caption.json
│   ├── DADA_2000/
│   │      ├── DADA_2000_VQA_annotation.json
│   │      └── DADA_2000_Dense_Caption.json
│   ├── DoTA/
│   │      ├── DoTA_VQA_annotation.json
│   │      └── DoTA_Dense_Caption.json
│   └── MANUAL_DATA/
│          ├── MANUAL_DATA_VQA_annotation.json
│          └── MANUAL_DATA_Dense_Caption.json
└── VRU_videos/ # total number of videos: 1000
    ├── CAP_DATA/
    │      └── VRU_1.mp4, ... , VRU_287.mp4
    ├── DADA_2000/
    │      └── VRU_1.mp4, ... , VRU_223.mp4
    ├── DoTA/
    │      └── VRU_1.mp4, ... , VRU_100.mp4
    └── MANUAL_DATA/
           └── VRU_1.mp4, ... , VRU_390.mp4
```

---


### 🤖 Model Response Generation

#### Open-Source Models

Run follow command to generate model responses:

```bash
python main.py \
  --task {VQA/Dense_Captioning} \
  --dataset {CAP_DATA/DADA_2000/DoTA/VRU_Accident} \
  --mode generation \
  --save_path ./Model_Response/{VQA/Dense_Captioning}/{Model_Name}/{Dataset_Name}_{Task}_response.json \
  --model InternVL3_8B
```

Examples:

```bash
python main.py \
  --task VQA \
  --dataset VRU_Accident \
  --mode generation \
  --save_path ./Model_Response/VQA/InternVL3_8B/VRU_Accident_VQA_response.json \
  --model InternVL3_8B
```
```bash
python main.py \
  --task Dense_Captioning \
  --dataset VRU_Accident \
  --mode generation \
  --save_path ./Model_Response/Dense_Captioning/InternVL3_8B/VRU_Accident_Dense_Captioning_response.json \
  --model InternVL3_8B
```

#### Closed-Source Models (API Key Required)
We provide the code to generate responses of GPT-4o-mini and Gemini-1.5-Flash.

```bash
python main.py \
  --task VQA \
  --dataset {CAP_DATA/DADA_2000/DoTA/VRU_Accident} \
  --mode generation \
  --save_path ./Model_Response/VQA/{GPT_4o_mini/Gemini_15_flash}/{Dataset_Name}_{Task}_response.json \
  --model InternVL3_8B
```

Examples:

```bash
python main.py \
  --task VQA \
  --dataset VRU_Accident \
  --mode generation \
  --save_path ./Model_Response/VQA/GPT_4o_mini/VRU_Accident_VQA_response.json \
  --model GPT_4o_mini \
  --api_key <Your_API_Key>
```

```bash
python main.py \
  --task VQA \
  --dataset VRU_Accident \
  --mode generation \
  --save_path ./Model_Response/VQA/Gemini_15_flash/VRU_Accident_VQA_response.json \
  --model Gemini_15_flash \
  --api_key <Your_API_Key>
```

---

### 🧪 Evaluation

#### VQA Evaluation


```bash
python main.py \
  --task VQA \
  --mode evaluation \
  --load_path ./Model_Response/VQA/<Model_Name> \
  --save_path ./results/VQA/<Model_Name>_VQA_Acc.json \
  --model <Model_Name> \
  --dataset All
```

Example:

```bash
python main.py \
  --task VQA \
  --mode evaluation \
  --load_path ./Model_Response/VQA/Qwen2_VL_7B \
  --save_path ./results/VQA/Qwen2_VL_7B_VQA_Acc.json \
  --model Qwen2_VL_7B \
  --dataset All
```

#### Dense Captioning Evaluation

```bash
  --task Dense_Captioning \
  --dataset All \
  --mode test \
  --model <Model_Name>
```

Example:

```bash
python main.py \
  --task Dense_Captioning \
  --dataset All \
  --mode test \
  --model InternVL25_4B
```


---

### 📦 Already generated model responses and results are available under:

- ./Model_Response/
- ./results/
---

### 📢 Acknowledgement
In addition to the 390 VRU-involved crash videos we newly collected, we also curated VRU-relevant videos from three publicly available accident datasets: [MM-AU](http://www.lotvsmmau.net/), [DADA-2000](https://github.com/JWFangit/LOTVS-DADA), and [DoTA](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly).

We annotated these videos with fine-grained Video Question Answering (VQA) and Dense Captioning labels as part of the VRU-Accident benchmark. We sincerely thank the original authors of these datasets for their efforts in collecting and sharing valuable crash video data with the research community.

