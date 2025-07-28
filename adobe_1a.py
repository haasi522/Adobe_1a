import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import torch
import joblib
import os
import json
import re
from transformers import BertTokenizer, BertModel

# Load classifier and label encoder
clf = joblib.load("pdf_classifier.pkl")
le = joblib.load("label_encoder.pkl")

# Load BERT (ensure it is downloaded)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=False)
bert = BertModel.from_pretrained("bert-base-uncased", local_files_only=False)
bert.eval()

def get_bert_embedding(text):
    with torch.no_grad():
        tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=30)
        output = bert(**tokens)
        return output.last_hidden_state[:, 0, :].squeeze().numpy()

def is_bold(span):
    return bool(span.get("flags", 0) & 2)  # font flag 2 = bold

def extract_features(pdf_path):
    doc = fitz.open(pdf_path)
    rows = []

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        line_items = []

        for block in blocks:
            for line in block.get("lines", []):
                full_text = ""
                font_sizes = []
                bold_flags = []
                word_lens = []

                spans = sorted(line["spans"], key=lambda x: x["bbox"][0])
                for span in spans:
                    text = span["text"].strip()
                    if not text:
                        continue

                    full_text += text + " "
                    font_sizes.append(span.get("size", 0))
                    word_lens += [len(w) for w in text.split()]
                    bold_flags.append(is_bold(span))

                text = full_text.strip()
                if not text:
                    continue

                avg_font_size = np.mean(font_sizes)
                avg_word_len = np.mean(word_lens) if word_lens else 0
                position_y = round(line["bbox"][1], 2)

                line_items.append({
                    "pdf_file": os.path.basename(pdf_path),
                    "page": page_num,
                    "text": text,
                    "font_size": round(avg_font_size, 2),
                    "is_bold": int(any(bold_flags)),
                    "is_uppercase": int(text.isupper()),
                    "is_page_1": int(page_num == 1),
                    "starts_with_number": int(bool(re.match(r"^(\d+[\.\)]?|[A-Z][\.\)]?)", text))),
                    "avg_word_len": round(avg_word_len, 2),
                    "position_y": position_y,
                    "num_words": len(text.split()),
                    "text_length": len(text),
                })

        # Sort top to bottom by Y-position
        line_items.sort(key=lambda x: -x["position_y"])

        # Merge patterns like '4.1.1' + 'Heading'
        merged = []
        i = 0
        while i < len(line_items):
            current = line_items[i]
            if (
                re.match(r"^(\d+(\.\d+)+|[A-Z]\.)$", current["text"].strip()) and
                i + 1 < len(line_items)
            ):
                nxt = line_items[i + 1]
                same_page = current["page"] == nxt["page"]
                vertical_gap = abs(current["position_y"] - nxt["position_y"])

                if same_page and vertical_gap < 10:
                    combined_text = f"{current['text']} {nxt['text']}"
                    merged_row = current.copy()
                    merged_row["text"] = combined_text
                    merged_row["text_length"] = len(combined_text)
                    merged_row["num_words"] = len(combined_text.split())
                    merged_row["avg_word_len"] = round(
                        np.mean([len(w) for w in combined_text.split()]), 2
                    )
                    merged.append(merged_row)
                    i += 2
                    continue

            merged.append(current)
            i += 1

        rows.extend(merged)

    return pd.DataFrame(rows)

def apply_title_rule(row):
    if row["is_page_1"] and row["font_size"] >= 22 and row["is_bold"] and row["text_length"] >= 5:
        return "TITLE"
    return row["predicted_label"]

def predict(df):
    embeddings = []
    for text in df["text"].tolist():
        try:
            emb = get_bert_embedding(text)
        except:
            emb = np.zeros(768)
        embeddings.append(emb)

    layout_cols = ["font_size", "is_uppercase", "is_page_1", "starts_with_number", "avg_word_len"]
    layout = df[layout_cols].values
    X = np.hstack([layout, embeddings])

    y_pred = clf.predict(X)
    df["predicted_label"] = le.inverse_transform(y_pred)
    df["label"] = df.apply(apply_title_rule, axis=1)

    df = df[df["label"].isin(["TITLE", "H1", "H2", "H3"])]
    return df[["label", "text", "page"]].to_dict(orient="records")

if __name__ == "__main__":
    input_dir = "/app/input"
    output_dir = "/app/output"
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            try:
                df = extract_features(pdf_path)
                results = predict(df)

                out_path = os.path.join(output_dir, filename.replace(".pdf", ".json"))
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2)

                print(f"✅ Processed: {filename}")

            except Exception as e:
                print(f"❌ Failed to process {filename}: {e}")

