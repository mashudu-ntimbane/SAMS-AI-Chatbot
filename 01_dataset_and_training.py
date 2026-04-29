"""
=============================================================
SAMS AI CHATBOT — Part 1: Dataset Creation & Model Training
=============================================================
Student Accommodation Management System (SAMS)
Author: Generated for SAMS Project
Compatible with: Google Colab (Python 3.10+)

HOW TO USE:
  Run this file in Google Colab. Each section is clearly marked.
  Install dependencies with the pip commands at the top.
=============================================================
"""

# ─────────────────────────────────────────────────────────────
# STEP 0 — Install Dependencies (run in Colab cell)
# ─────────────────────────────────────────────────────────────
# Uncomment these lines when running in Google Colab:
#
# !pip install transformers datasets torch scikit-learn seaborn matplotlib
# !pip install flask flask-cors accelerate


# ─────────────────────────────────────────────────────────────
# STEP 1 — Imports
# ─────────────────────────────────────────────────────────────
import json
import random
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)

# ─────────────────────────────────────────────────────────────
# STEP 2 — Synthetic Dataset Creation
#
# We define 5 intents covering common student queries.
# Each intent has:
#   - Multiple phrasing variations (helps model generalise)
#   - A list of canned responses to randomly select from
#
# You can expand this dataset easily — just add more patterns.
# ─────────────────────────────────────────────────────────────

INTENTS = {
    "payment_query": {
        "patterns": [
            "How do I pay my rent?",
            "What are the payment options available?",
            "When is rent due?",
            "Can I pay my accommodation fees online?",
            "My payment was declined, what should I do?",
            "Do you accept credit card payments?",
            "How much is the monthly fee?",
            "I missed my rent payment, what happens?",
            "Can I set up a direct debit for rent?",
            "What is the late payment penalty?",
            "Is there a payment plan available?",
            "I want to know about payment methods",
            "How can I pay accommodation fees?",
            "Where do I pay my rent?",
            "Is there a student payment portal?",
            "Can I pay in instalments?",
            "What happens if I don't pay rent on time?",
            "Do you accept EFT payments?",
            "When will my payment reflect?",
            "I need to make a payment for my room",
        ],
        "responses": [
            "Rent is due on the 1st of each month. You can pay via the SAMS student portal using EFT, credit card, or debit card. Set up a monthly debit order to never miss a payment!",
            "We accept EFT, credit/debit cards, and cash payments at the admin office. Payments are due by the 1st of each month. Late payments attract a 5% penalty.",
            "You can pay your accommodation fees through the SAMS portal at sams.ac.za/pay, or visit the finance office in Block A. A payment plan may be arranged — contact finance@sams.ac.za.",
        ]
    },

    "rules_query": {
        "patterns": [
            "What are the accommodation rules?",
            "Can I smoke in my room?",
            "What time is quiet hours?",
            "Is alcohol allowed in the residence?",
            "Can I have guests overnight?",
            "What are the noise rules?",
            "Are pets allowed?",
            "What happens if I break the rules?",
            "Is cooking allowed in the room?",
            "What are the residence regulations?",
            "Can I play music in my room?",
            "What is the curfew time?",
            "Am I allowed to have a mini-fridge?",
            "Are candles or open flames permitted?",
            "What is the dress code in common areas?",
            "Can I rearrange the furniture in my room?",
            "What electrical appliances are allowed?",
            "Is there a laundry schedule?",
            "What are the cleaning rules for common areas?",
            "Can I decorate my room?",
        ],
        "responses": [
            "Smoking is strictly prohibited in all indoor areas. Quiet hours are 10pm–7am on weekdays and midnight–8am on weekends. Alcohol is permitted in private rooms for students over 18 but not in common areas.",
            "Pets are not allowed. Cooking appliances (except small kettles and approved microwaves) are restricted to kitchen areas. Rule violations may result in a warning, fine, or eviction.",
            "Full residence rules are available on the SAMS portal under 'Residence Handbook'. Key rules include: no smoking indoors, respect quiet hours, no overnight guests without prior approval, and keep common areas clean.",
        ]
    },

    "visitor_policy": {
        "patterns": [
            "Can I have visitors?",
            "What is the visitor policy?",
            "Are guests allowed in my room?",
            "Can my friend sleep over?",
            "How do I register a visitor?",
            "What are the visiting hours?",
            "Can I have overnight guests?",
            "Is there a limit to how many visitors I can have?",
            "Do visitors need to sign in?",
            "Can family members visit me?",
            "My friend wants to visit, what must they do?",
            "Are visitors allowed in the residence?",
            "How long can a visitor stay?",
            "Can visitors come into the building after hours?",
            "Do guests need to show ID at reception?",
            "What is the guest registration process?",
            "Are there days when visitors are not allowed?",
            "Can I invite someone to stay the weekend?",
            "Is there a visitor fee?",
            "Where can guests park when visiting?",
        ],
        "responses": [
            "Visitors are welcome between 8am and 10pm daily. Overnight guests require a written request submitted 24 hours in advance via the SAMS portal. Each student may have a maximum of 2 registered visitors at a time.",
            "All visitors must sign in at reception and show a valid ID. Visiting hours are 8am–10pm. Overnight stays must be approved by the residence manager — apply through the SAMS portal under 'Visitor Request'.",
            "Guests can visit during approved hours (8am–10pm). To register an overnight visitor, submit a request on the SAMS portal at least 24 hours ahead. Unapproved overnight guests may result in a disciplinary notice.",
        ]
    },

    "application_status": {
        "patterns": [
            "What is the status of my application?",
            "Has my accommodation application been approved?",
            "When will I hear back about my application?",
            "I submitted my application, what happens next?",
            "How long does the application process take?",
            "My application is still pending, is this normal?",
            "How do I check if my application was successful?",
            "I haven't received a response to my application",
            "Can I track my application status online?",
            "What documents are needed for my application?",
            "I need to update my application details",
            "Can I apply for a room change?",
            "When do applications open for next year?",
            "How will I know if I got accommodation?",
            "What is the waiting list for accommodation?",
            "My application was rejected, what can I do?",
            "Can I appeal an accommodation decision?",
            "I applied weeks ago and heard nothing",
            "Is my application still being processed?",
            "How do I complete my application?",
        ],
        "responses": [
            "You can check your application status anytime on the SAMS portal under 'My Application'. Processing takes 5–10 business days. You'll also receive an email notification when a decision is made.",
            "Log into your SAMS account and click 'Application Status' to see real-time updates. If your status shows 'Under Review' for more than 10 business days, contact admissions@sams.ac.za.",
            "Application results are sent by email and visible on the SAMS portal. If your application is pending, it is still being reviewed. Appeals for rejected applications must be submitted within 14 days via the portal.",
        ]
    },

    "general_enquiry": {
        "patterns": [
            "Hello",
            "Hi there",
            "Good morning",
            "I need help",
            "Can you help me?",
            "What can you do?",
            "Who are you?",
            "Tell me about SAMS",
            "What services do you offer?",
            "I have a question",
            "Help",
            "Hey",
            "What is SAMS?",
            "I'm a new student",
            "I need information",
            "How does this work?",
            "Is anyone there?",
            "I have a complaint",
            "I want to speak to a person",
            "Thank you",
        ],
        "responses": [
            "Hi! I'm the SAMS virtual assistant. I can help you with payment queries, accommodation rules, visitor policies, and application status. What can I help you with today?",
            "Hello! Welcome to SAMS. I'm here to assist with your accommodation questions. Try asking about payments, rules, visitors, or your application status.",
            "I'm your SAMS chatbot assistant! For urgent issues or complaints, please contact our support team at support@sams.ac.za or visit the admin office in Block A.",
        ]
    }
}


def build_dataframe(intents_dict: dict) -> pd.DataFrame:
    """
    Convert the intents dictionary into a flat DataFrame.
    Each row = one training example with columns: text, intent.

    Args:
        intents_dict: Dictionary of intents with patterns and responses.

    Returns:
        A shuffled pandas DataFrame ready for model training.
    """
    rows = []
    for intent_name, intent_data in intents_dict.items():
        for pattern in intent_data["patterns"]:
            rows.append({"text": pattern, "intent": intent_name})

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    print(f"Total training examples: {len(df)}")
    print("\nIntent distribution:")
    print(df["intent"].value_counts())
    return df


# Build the dataset
df = build_dataframe(INTENTS)


# ─────────────────────────────────────────────────────────────
# STEP 3 — Text Preprocessing
#
# We clean text before feeding it to the model.
# DistilBERT handles most NLP internally, but basic cleaning
# helps with edge cases (typos, extra spaces, casing).
# ─────────────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Clean and normalise raw input text.

    Steps:
      1. Lowercase everything
      2. Remove special characters (keep letters, numbers, spaces)
      3. Collapse multiple spaces into one
      4. Strip leading/trailing whitespace

    Args:
        text: Raw user input string.

    Returns:
        Cleaned text string.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()   # collapse spaces
    return text


# Apply preprocessing to dataset
df["text_clean"] = df["text"].apply(preprocess_text)

print("\nSample cleaned texts:")
print(df[["text", "text_clean"]].head(5).to_string())


# ─────────────────────────────────────────────────────────────
# STEP 4 — Label Encoding
#
# PyTorch needs integer labels, not strings.
# LabelEncoder maps: "payment_query" → 0, "rules_query" → 1, etc.
# We save the mapping so we can decode predictions later.
# ─────────────────────────────────────────────────────────────

label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["intent"])

# Save the mapping for inference
id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
label2id = {label: i for i, label in id2label.items()}

print("\nLabel mapping:")
for idx, intent in id2label.items():
    print(f"  {idx} → {intent}")


# ─────────────────────────────────────────────────────────────
# STEP 5 — Train / Test Split
# ─────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    df["text_clean"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"]  # ensure equal class representation
)

print(f"\nTraining set: {len(X_train)} examples")
print(f"Test set:     {len(X_test)} examples")


# ─────────────────────────────────────────────────────────────
# STEP 6 — Tokenisation with DistilBERT
#
# DistilBERT tokenizer converts text → token IDs + attention masks.
# max_length=64 is enough for short queries (saves memory).
# ─────────────────────────────────────────────────────────────

MODEL_NAME = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)


class IntentDataset(Dataset):
    """
    PyTorch Dataset wrapper for intent classification.

    Each item returns:
      - input_ids:      token IDs for the text
      - attention_mask: 1 for real tokens, 0 for padding
      - labels:         integer class label
    """

    def __init__(self, texts: list, labels: list, tokenizer, max_length: int = 64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long)
        }


# Create Dataset objects
train_dataset = IntentDataset(X_train, y_train, tokenizer)
test_dataset  = IntentDataset(X_test,  y_test,  tokenizer)

# Create DataLoaders (batches data for GPU efficiency)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=8, shuffle=False)

print(f"\nBatches per epoch: {len(train_loader)}")


# ─────────────────────────────────────────────────────────────
# STEP 7 — Model Initialisation
#
# We load a pre-trained DistilBERT and add a classification head
# on top (DistilBertForSequenceClassification does this for us).
# Fine-tuning adjusts the entire model on our domain-specific data.
# ─────────────────────────────────────────────────────────────

NUM_CLASSES = len(label_encoder.classes_)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
    id2label=id2label,
    label2id=label2id
)
model.to(device)

print(f"Model loaded: {MODEL_NAME}")
print(f"Output classes: {NUM_CLASSES}")


# ─────────────────────────────────────────────────────────────
# STEP 8 — Training Loop
#
# We use AdamW (Adam with weight decay) — standard for Transformers.
# A linear learning rate scheduler warms up then decays the LR.
# ─────────────────────────────────────────────────────────────

EPOCHS      = 10
LR          = 2e-5   # standard fine-tuning learning rate for BERT
WARMUP_FRAC = 0.1    # 10% of steps used for warmup

total_steps    = len(train_loader) * EPOCHS
warmup_steps   = int(total_steps * WARMUP_FRAC)

optimizer  = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler  = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# Track metrics for plotting
history = {"train_loss": [], "val_accuracy": []}


def evaluate_model(model, loader, device):
    """
    Run model on a DataLoader and return accuracy + all predictions.

    Args:
        model:  Trained PyTorch model.
        loader: DataLoader for evaluation.
        device: CUDA or CPU.

    Returns:
        Tuple of (accuracy, all_predictions, all_true_labels)
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits
            preds   = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, all_preds, all_labels


# ── Main Training Loop ────────────────────────────────────────
print("\n" + "="*55)
print(" Starting Training ")
print("="*55)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    val_acc, _, _ = evaluate_model(model, test_loader, device)

    history["train_loss"].append(avg_loss)
    history["val_accuracy"].append(val_acc)

    print(f"Epoch {epoch+1:02d}/{EPOCHS}  |  Loss: {avg_loss:.4f}  |  Val Acc: {val_acc:.4f}")

print("\n✓ Training complete!")


# ─────────────────────────────────────────────────────────────
# STEP 9 — Evaluation & Metrics
# ─────────────────────────────────────────────────────────────

_, y_pred, y_true = evaluate_model(model, test_loader, device)
intent_names = label_encoder.classes_

print("\n" + "="*55)
print(" Classification Report ")
print("="*55)
print(classification_report(y_true, y_pred, target_names=intent_names))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=intent_names,
    yticklabels=intent_names
)
plt.title("SAMS Chatbot — Confusion Matrix")
plt.xlabel("Predicted Intent")
plt.ylabel("True Intent")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# Training history plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history["train_loss"], marker="o", color="steelblue")
ax1.set_title("Training Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True, alpha=0.3)

ax2.plot(history["val_accuracy"], marker="o", color="seagreen")
ax2.set_title("Validation Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_ylim([0, 1.05])
ax2.grid(True, alpha=0.3)

plt.suptitle("SAMS Chatbot Training Metrics")
plt.tight_layout()
plt.savefig("training_metrics.png", dpi=150)
plt.show()


# ─────────────────────────────────────────────────────────────
# STEP 10 — Save Model & Artefacts
#
# Save everything needed for inference:
#   - model weights
#   - tokenizer
#   - label mapping
#   - intent responses
# ─────────────────────────────────────────────────────────────

SAVE_DIR = "./sams_chatbot_model"

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

# Save label mapping as JSON
with open(f"{SAVE_DIR}/label_mapping.json", "w") as f:
    json.dump({"id2label": id2label, "label2id": label2id}, f, indent=2)

# Save responses for inference
responses_map = {
    intent: data["responses"] for intent, data in INTENTS.items()
}
with open(f"{SAVE_DIR}/responses.json", "w") as f:
    json.dump(responses_map, f, indent=2)

print(f"\n✓ Model and artefacts saved to: {SAVE_DIR}")
print("Files saved:")
import os
for fname in os.listdir(SAVE_DIR):
    print(f"  {fname}")
