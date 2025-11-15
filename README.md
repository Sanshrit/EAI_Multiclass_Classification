# ANLI – Natural Language Inference (3-Class Classification)

The repository implements an end-to-end machine learning pipeline for the ANLI (Adversarial Natural Language Inference) Round 2 dataset.

The task is a three-way classification:
- entailment
- neutral
- contradiction

The project includes:
- Data loading and cleaning
- Exploratory Data Analysis (EDA)
- Baseline model (TF-IDF + Logistic Regression)
- Transformer fine-tuning (DeBERTa-v3-base)
- Evaluation on dev and test sets
- Model hosting on Hugging Face
- Dockerized inference API using FastAPI
- Documentation for full reproducibility

## Dataset Overview
Dataset source: [Dataset](https://huggingface.co/datasets/facebook/anli)

Splits used:
- train_r2: 45,460 examples
- dev_r2: 1,000 examples
- test_r2: 1,000 examples

Each sample contains:
| **Field** | **Description**
|---|---|
| premise | Passage of Text
| hypothesis | Text to compare against the premise
| label | entailment / neutral / contradiction
| reason | Human-written explanation

Duplicate Handling:

31 duplicate (premise, hypothesis) pairs were found.
All had identical labels; duplicates were removed.
Cleaned training set size: **45,429**.

## Exploratory Data Analysis
Class Distribution
- neutral: 46.1%
- entailment: 31.8%
- contradiction: 22.1%

Macro F1 is used to address this imbalance.

Length Statistics
- Premise average length: 54 tokens
- Hypothesis average length: 10 tokens
- 95th percentile fits within 256 tokens

Maximum sequence length was set to **256**.

## Baseline Model: TF-IDF + Logistic Regression
Configuration:
```
TfidfVectorizer(
    min_df=3,
    max_features=100000,
    ngram_range=(1, 2),
    sublinear_tf=True
)
```
The baseline model performs confirms that ANLI-R2 is adversarial and challenging.

## Transformer Model: DeBERTa-v3-base
Model: **microsoft/deberta-v3-base**

Reasons for selecting DeBERTa-v3:
- Strong performance on NLI benchmarks
- Enhanced attention and disentangled position encoding
- Better contextual representations than BERT/RoBERTa

### Training Setup
| **Training setup** | **Value**
|---|---|
| Max length | 256
| Epochs | 4
| Batch size | 16
| Optimizer | AdamW
| Learning Rate | 2e-5
| Warmup ratio | 0.06
| Early stopping | patience 2
| Evaluation Metric | macro F1

### Evaluation Results
Dev Set:
- Accuracy: **49.4**%
- Macro F1: **49.2**%

Test Set:
- Accuracy: **50.1**%
- Macro F1: **50.0**%

## Model Hosting

The fine-tuned model is hosted on Hugging Face:

[DeBERTa hosted on HF](https://huggingface.co/BakshiSan/deberta-v3-anli-r2)

The Docker container downloads the model automatically at runtime.


##  Dockerized Inference API
FastAPI server provides an inference endpoint for real-time predictions.

Build the Docker Image
``` 
docker build -t anli-r2-api .
```

Run the container
```
docker run --rm -p 8000:8000 anli-r2-api
```
This loads the model from HuggingFace and starts the API server.

### Test the API
Prediction
```
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"premise": "A man is playing guitar.", "hypothesis": "A person is making music."}'
```

Sample Output:
```
{
  "label_id": 0,
  "label": "entailment",
  "probabilities": {
    "entailment": 0.9993,
    "neutral": 0.00027,
    "contradiction": 0.00037
  }
}
```

## Reproducibility Guide

1.  Clone the repository
    ```bash
    git clone https://github.com/Sanshrit/EAI_Multiclass_Classification.git
    ```

2.  Run the notebook at:
    `notebooks/01_anli_r2_eda_and_model.ipynb`

    This performs:
    * **EDA** (Exploratory Data Analysis)
    * Baseline modeling
    * Data cleaning
    * Transformer fine-tuning
    * Upload to HuggingFace

3. Deploy the API via Docker
```bash
docker build -t anli-r2-api .
docker run -p 8000:8000 anli-r2-api
```

