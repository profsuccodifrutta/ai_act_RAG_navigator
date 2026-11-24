# AI Act RAG Navigator

A risk-aware Retrieval-Augmented Generation (RAG) assistant for the EU AI Act (Regulation (EU) 2024/1689).

You describe your AI product and ask a compliance question. The system first classifies the product into one of the four AI Act risk tiers, then uses that label, together with your role (provider, deployer, and so on), to steer retrieval over the text of the regulation. The LLM answers only from the retrieved articles and cites them.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/profsuccodifrutta/ai_act_RAG_navigator/blob/main/NaN_project.ipynb)

Live demo: [AI Act Advisor on Hugging Face Spaces](https://huggingface.co/spaces/profsuccofrutta/ai-act-advisor) (requires a Hugging Face access token, see [Using the live demo](#using-the-live-demo)).

Built as the project for the Data Visualization and Text Mining course in the MSc in Data Science.

## Why

General-purpose LLMs aren't aligned with regulatory frameworks. On the AI Act they can sound confident and still misread the risk tier or invent obligations, which makes them unreliable for compliance work. We built this system to keep the model inside the actual text of the regulation. A classifier works out which rules apply, retrieval narrows the context to the relevant provisions, and every answer points back to the articles it relies on.

## How it works

```
 user's product description ──► risk classifier (DistilBERT)
                                        │
             ┌──────────────────────────┴─────────────────────────┐
     Unacceptable_Risk                                   High / Limited / Minimal
             │                                                    │
    guardrail: fixed "prohibited"              hybrid retrieval (BM25 + FAISS, RRF)
    alert, no LLM call                                            │
                                         metadata re-ranking on role + risk level
                                                                  │
                                         prompt with retrieved chunks ──► Gemma-2-2B-it
                                                                  │
                                                  answer with article citations
```

### 1. Parsing the regulation

The AI Act PDF is cleaned of headers, footnotes and page noise, then split into recitals, articles and annexes.

### 2. Chunking and metadata enrichment

Each legal unit is split into overlapping chunks tagged with its article or annex reference. A spaCy pipeline with custom rules adds entities such as legal roles and risk levels as metadata.

### 3. Exploratory analysis

Charts of chunk lengths, category counts, entity frequencies and TF-IDF terms check that the parsing went as expected.

### 4. Embeddings and vector store

Chunks are embedded with the multilingual BGE-M3 model and stored in a FAISS index.

### 5. Retrieval

A LangChain `EnsembleRetriever` runs BM25 (weight 0.3) and FAISS (weight 0.7) side by side and merges them with Reciprocal Rank Fusion.

On top of that, `retrieve_and_filter_chunks` fetches 60 candidates, rebuilds a min-max normalized RRF score from each one's rank, and blends it with a metadata score based on fuzzy matching of role and risk level:

`final = 0.8 · semantic + 0.2 · metadata`

Chunks scoring under 40% of the top chunk are dropped, and the best 5 are kept.

This matters for questions like "how long must I keep system logs?" asked by a high-risk provider. Plain semantic search puts Article 26 first, but Article 26 covers deployers. With re-ranking, the provider obligations in Articles 18, 19 and 22 move to the top.

### 6. Risk classifier

The classifier is trained on a synthetic dataset generated with Gemini (`ai_risk_dataset2.csv.txt`, about 980 product descriptions balanced across the four labels). Every model uses the same stratified 80/10/10 split.

| Model | Accuracy | F1 macro | Recall macro | ROC AUC |
|---|---|---|---|---|
| DistilBERT (selected) | **96.94%** | **0.9696** | **0.9696** | 0.9939 |
| Legal-BERT | 96.94% | 0.9692 | 0.9692 | **0.9944** |
| Random Forest (TF-IDF) | 92.86% | 0.9292 | 0.9279 | 0.9923 |
| Linear SVM (TF-IDF) | 92.86% | 0.9284 | 0.9283 | 0.9859 |
| ULMFiT (AWD-LSTM, domain-adapted on the AI Act) | 84.69% | 0.8458 | 0.8475 | 0.9811 |
| XGBoost (TF-IDF) | 83.67% | 0.8386 | 0.8367 | 0.9629 |
| BiGRU + GloVe 100d | 63.27% | 0.6200 | 0.6312 | 0.9079 |

The no-information rate is 25.51%. DistilBERT matches Legal-BERT with about 60% of the parameters, so the pipeline uses DistilBERT.

![Training and validation loss per epoch for DistilBERT, Legal-BERT, ULMFiT and BiGRU](assets/loss_curves.png)

Both BERT models converge within three epochs. After that, training loss keeps falling toward zero while validation loss levels off around 0.1 (DistilBERT) and 0.05 (Legal-BERT). The ULMFiT curve covers the six epochs of gradual unfreezing, and its validation loss stops improving at about 0.57. The BiGRU learns slowly: after ten epochs its validation loss is still 0.77. The curves come from the training logs saved in the notebook, so no model was retrained to draw them.

### 7. Generation

Answers come from google/gemma-2-2b-it, run locally through a Hugging Face pipeline with temperature 0.1 and up to 512 new tokens. `build_super_prompt()` writes the user's risk tier and role into the system instructions, tells the model to use only the retrieved context, and requires article citations.

When the classifier returns `Unacceptable_Risk`, the pipeline stops there and returns a fixed alert without calling the LLM.

### 8. Evaluation

The notebook ends with three case studies. Each one is also run through the same LLM without RAG for comparison.

| Case | Plain LLM | This system |
|---|---|---|
| Emotion recognition at work and school | Calls it high-risk and lists compliance steps | Classified as unacceptable; the guardrail stops it |
| E-commerce FAQ chatbot | Says a QMS and technical docs are required | Classified as limited risk; no high-risk obligations, cites Art. 17 and Annex VII |
| CE marking for high-risk systems | Says CE marking isn't required | Correct, cites Art. 48 and Art. 49(1) |

## Repository contents

| File | What it is |
|---|---|
| `NaN_project.ipynb` | The full pipeline: parsing, EDA, NER, embeddings, retrieval, classifiers, generation, evaluation |
| `ai_act_2024.pdf` | Official text of the EU AI Act (Regulation 2024/1689) |
| `ai_risk_dataset2.csv.txt` | Synthetic training set for the risk classifier (`text`, `label`) |
| `pyproject.toml`, `uv.lock` | Dependencies, managed with `uv` |
| `assets/loss_curves.png` | Loss curves shown above |

Running the notebook also creates these files, which are not committed: `data/polished_chunks.json`, `ai_act_rag_index/`, `distilBERT_classifier/`, `legalBERT_classifier/`, `ulmfit_ai_risk_classifier.pkl`, and the GloVe files.

## Using the live demo

The app generates answers through the Hugging Face Inference API, so it asks for your own access token. Create one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (the "Make calls to Inference Providers" permission is enough) and paste it into the token field. The token is only used for your request and is never stored.

## Running the notebook

The notebook targets Google Colab with a GPU runtime, since embedding and fine-tuning use CUDA.

1. Open the notebook with the Colab badge above.
2. Accept the license for [`google/gemma-2-2b-it`](https://huggingface.co/google/gemma-2-2b-it) on Hugging Face, create an access token, and save it in Colab Secrets as `hf_gemma`.
3. Run all cells. The first ones clone this repo, install `uv` and the dependencies, and load the PDF and dataset from the repo root.

You can run it locally too, but you'll have to remove the `google.colab` imports (Drive and `userdata`) and pass the Hugging Face token some other way.

## Limitations

The classifier is trained on synthetic data. Real product descriptions tend to be vaguer, and labels written by an LLM may be wrong on borderline legal cases. Generation is evaluated qualitatively on three case studies, not on a benchmark. Gemma-2-2B is a small model, so its answers stay grounded but can be shallow.

This is a course project and not legal advice.

## Tech stack

PyMuPDF, spaCy, LangChain, FAISS, rank-bm25, sentence-transformers (BGE-M3), Hugging Face Transformers, fastai, TensorFlow/Keras, scikit-learn, XGBoost, Plotly.
