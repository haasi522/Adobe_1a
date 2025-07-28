
# Adobe_1A – PDF Heading Classification (Dockerized)

## Project Overview

###  Mission

In this challenge, the task is to transform an unstructured PDF into a meaningful, machine-readable document outline. Rather than just extracting raw text, the goal is to **understand and organize the logical structure** of the document—identifying its **Title**, and the hierarchical **headings (H1, H2, H3)**.

This structured outline serves as the foundation for the next stages of the hackathon. It enables downstream tasks such as content understanding, document summarization, or intelligent navigation through long technical documents.

### Objective

Given a PDF file, the system should:

- Analyze each line of text
- Extract layout and font-based features (e.g., font size, boldness, position)
- Generate semantic embeddings using a BERT-based model
- Classify each line into one of the following categories:
  - `TITLE`
  - `H1` (Top-level heading)
  - `H2` (Sub-heading)
  - `H3` (Sub-sub-heading)
  - `OTHER` (Body text or non-heading content)
- Produce a **clean, hierarchical outline** of the document structure

---

## Approach

The pipeline combines both **layout-based features** and **semantic embeddings** to classify each line of text from the PDF. The key stages include:

1. **Text and Feature Extraction**  
   Using PyMuPDF (`fitz`), we extract:
   - Font size
   - Boldness
   - Y-coordinate (position on page)
   - Word count, average word length
   - Whether the text starts with a number or is uppercase

2. **Semantic Understanding with BERT**  
   Each line is embedded using a BERT model (`bert-base-uncased`) to capture its semantic meaning.

3. **Classification**  
   A trained scikit-learn classifier (`pdf_classifier.pkl`) predicts the appropriate heading level.

4. **Output Generation**  
   The predictions are saved in a structured JSON or CSV format representing the document's outline.

---

## Libraries and Tools Used

- **PyMuPDF (fitz)** – For reading PDFs and extracting layout features
- **Transformers (BERT)** – For generating semantic embeddings
- **scikit-learn** – For classifier training and inference
- **NumPy / pandas** – For data preprocessing and analysis
- **Docker** – For containerized deployment and reproducibility

---

## Files and Project Structure

This repository contains the following key components:

- `adobe_1a.py`  
  Main script that handles PDF parsing, feature extraction, BERT embedding, and classification.

- `Dockerfile`  
  Defines the environment and dependencies to run the solution in a Docker container.

- `requirements.txt`  
  Python dependencies needed inside the container.

- `model.safetensors`  
  Pretrained model weights (included inside the Docker image to avoid large file uploads to GitHub).

- `pdf_classifier.pkl`  
  Trained classifier model to predict heading labels based on extracted features.

- `label_encoder.pkl`  
  Scikit-learn label encoder for mapping predicted classes to readable labels.

- `bert_embeddings.npy`  
  Optional precomputed BERT embeddings to speed up repeated runs.

- `output/`  
  Stores the final JSON/CSV predictions per document. This folder is excluded from Git using `.gitignore`.

- `.gitignore`  
  Ensures large files and temporary outputs are not committed to version control.

- `README.md`  
  Project documentation and execution instructions.

---
.
