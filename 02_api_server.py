"""
=============================================================
SAMS AI CHATBOT — Part 2: Hybrid Approach + Flask API
=============================================================
This file covers:
  A) Hybrid intent matching (keywords + ML fallback)
  B) Chatbot response logic
  C) Flask API server with POST /chat endpoint

Run AFTER Part 1 has trained and saved the model.
=============================================================
"""

import json
import os
import re
import random
import logging
from datetime import datetime
from typing import Tuple, Optional

import torch
import numpy as np
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)

# Flask imports
from flask import Flask, request, jsonify
from flask_cors import CORS


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

MODEL_DIR = "C:/xampp/htdocs/New folder/Chatbot/sams model" # path where model was saved
CONFIDENCE_THRESHOLD = 0.60  # below this, fall back to rules or "unknown"
MAX_INPUT_LENGTH     = 200   # reject suspiciously long inputs
LOG_FILE             = "query_log.jsonl"     # log user queries for future training

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# APPROACH B — Hybrid: Keyword Rules + ML Fallback
#
# How it works:
#   1. First, try keyword matching (fast, interpretable)
#   2. If keyword match found → return intent directly
#   3. If no keyword match → run DistilBERT classifier
#   4. If classifier confidence < threshold → return "unknown"
#
# When to use each:
#   • Keywords: great for exact phrases, acronyms, common words
#   • ML:       handles paraphrasing, spelling variants, new phrasing
# ─────────────────────────────────────────────────────────────

KEYWORD_RULES = {
    "payment_query": [
        "pay", "payment", "rent", "fee", "cost", "debit",
        "invoice", "overdue", "credit card", "eft", "direct debit",
        "installment", "installment", "penalty", "late payment"
    ],
    "rules_query": [
        "rule", "regulation", "policy", "allowed", "permitted",
        "smoking", "alcohol", "noise", "quiet", "pet", "curfew",
        "candle", "cooking", "appliance", "laundry"
    ],
    "visitor_policy": [
        "visitor", "guest", "visit", "overnight", "sleepover",
        "sign in", "reception", "friend", "family", "hours"
    ],
    "application_status": [
        "application", "apply", "applied", "status", "approved",
        "rejected", "pending", "waiting list", "appeal",
        "processing", "hear back", "decision"
    ],
    "general_enquiry": [
        "hello", "hi", "help", "thanks", "thank you", "who are you",
        "what can you do", "goodbye", "bye", "morning", "evening"
    ]
}


def keyword_match(text: str) -> Optional[str]:
    """
    Match text against keyword rules.

    Scans the cleaned input for keyword phrases.
    Returns the first matching intent, or None if no match.

    Args:
        text: Preprocessed (lowercase) user query.

    Returns:
        Intent name string, or None.
    """
    for intent, keywords in KEYWORD_RULES.items():
        for kw in keywords:
            if kw in text:
                logger.debug(f"Keyword match: '{kw}' → {intent}")
                return intent
    return None


# ─────────────────────────────────────────────────────────────
# MODEL INFERENCE CLASS
# ─────────────────────────────────────────────────────────────

class SAMSChatbot:
    """
    SAMS AI Chatbot — hybrid intent classifier and responder.

    Loads a fine-tuned DistilBERT model and uses it alongside
    keyword rules to classify student queries and return
    appropriate responses.
    """

    def __init__(self, model_dir: str):
        """
        Initialise the chatbot by loading all saved artefacts.

        Args:
            model_dir: Directory where model was saved in Part 1.
        """
        logger.info(f"Loading SAMS chatbot from: {model_dir}")

        # Detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Running on: {self.device}")

        # Load tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        self.model     = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        # Load label mapping
        with open(os.path.join(model_dir, "label_mapping.json")) as f:
            mapping = json.load(f)
        self.id2label = {int(k): v for k, v in mapping["id2label"].items()}
        self.label2id = mapping["label2id"]

        # Load response templates
        with open(os.path.join(model_dir, "responses.json")) as f:
            self.responses = json.load(f)

        logger.info(f"Chatbot ready. Intents: {list(self.label2id.keys())}")

    def preprocess(self, text: str) -> str:
        """
        Clean and normalise raw text input.

        Args:
            text: Raw user message.

        Returns:
            Cleaned, lowercase text.
        """
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def classify_with_ml(self, text: str) -> Tuple[str, float]:
        """
        Run the DistilBERT model on cleaned text.

        Args:
            text: Preprocessed user query.

        Returns:
            Tuple of (predicted_intent, confidence_score).
        """
        # Tokenise
        encoding = self.tokenizer(
            text,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Inference (no gradient calculation needed)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits
            probs   = torch.softmax(logits, dim=-1).squeeze()

        # Get top prediction
        top_class      = torch.argmax(probs).item()
        confidence     = probs[top_class].item()
        predicted_intent = self.id2label[top_class]

        logger.debug(f"ML prediction: {predicted_intent} ({confidence:.3f})")
        return predicted_intent, confidence

    def get_response(self, intent: str) -> str:
        """
        Select a random response for a given intent.

        Args:
            intent: Classified intent name.

        Returns:
            A response string.
        """
        if intent in self.responses:
            return random.choice(self.responses[intent])
        return (
            "I'm not sure I understand that. Could you rephrase your question? "
            "You can ask about payments, accommodation rules, visitors, or your application status."
        )

    def chat(self, user_message: str, use_hybrid: bool = True) -> dict:
        """
        Main chatbot method — classify intent and generate response.

        Hybrid flow:
          1. Preprocess text
          2. Try keyword matching (if use_hybrid=True)
          3. Fall back to DistilBERT if no keyword match
          4. Fall back to "unknown" if ML confidence is too low
          5. Return response + metadata

        Args:
            user_message: Raw text from the student.
            use_hybrid:   Whether to use keyword rules before ML.

        Returns:
            Dictionary with intent, confidence, response, and method used.
        """
        # Validate input
        if not user_message or not user_message.strip():
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "response": "Please type a message and I'll do my best to help you.",
                "method": "validation"
            }

        if len(user_message) > MAX_INPUT_LENGTH:
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "response": "Your message is too long. Please keep queries concise.",
                "method": "validation"
            }

        # Preprocess
        clean_text = self.preprocess(user_message)
        method     = "unknown"
        intent     = None
        confidence = 0.0

        # Step 1: Keyword matching
        if use_hybrid:
            matched_intent = keyword_match(clean_text)
            if matched_intent:
                intent     = matched_intent
                confidence = 1.0   # keyword matches are definitive
                method     = "keyword"

        # Step 2: ML fallback
        if intent is None:
            intent, confidence = self.classify_with_ml(clean_text)
            method = "ml"

            # Step 3: Low-confidence fallback
            if confidence < CONFIDENCE_THRESHOLD:
                intent     = "unknown"
                method     = "fallback"

        # Get response
        response = self.get_response(intent)

        # Log query for future training data collection
        self._log_query(user_message, clean_text, intent, confidence, method)

        return {
            "intent":     intent,
            "confidence": round(float(confidence), 4),
            "response":   response,
            "method":     method
        }

    def _log_query(self, raw: str, clean: str, intent: str, confidence: float, method: str):
        """
        Append query to a JSONL log file for future training.

        Each line is a valid JSON object — easy to re-train on later.

        Args:
            raw:        Original user message.
            clean:      Preprocessed message.
            intent:     Predicted intent.
            confidence: Model confidence.
            method:     Which method classified this query.
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "raw_text":  raw,
            "clean_text": clean,
            "intent":    intent,
            "confidence": round(float(confidence), 4),
            "method":    method
        }
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


# ─────────────────────────────────────────────────────────────
# FLASK API — POST /chat
#
# This creates a lightweight REST API that:
#   - Accepts JSON: {"message": "user query"}
#   - Returns JSON: {"intent": ..., "response": ..., ...}
#
# Your PHP backend (or any frontend) calls this endpoint.
# ─────────────────────────────────────────────────────────────

# Load chatbot (global singleton — loaded once on startup)
chatbot = SAMSChatbot(MODEL_DIR)

# Initialise Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (needed for PHP integration)


@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint — useful for monitoring.

    Returns:
        JSON with status "ok" and model info.
    """
    return jsonify({
        "status":  "ok",
        "service": "SAMS Chatbot API",
        "version": "1.0.0"
    })


@app.route("/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint.

    Expected request body (JSON):
        {
            "message":     "How do I pay my rent?",
            "use_hybrid":  true    (optional, default: true)
        }

    Returns (JSON):
        {
            "success":    true,
            "intent":     "payment_query",
            "confidence": 0.9821,
            "response":   "Rent is due on the 1st...",
            "method":     "keyword"
        }
    """
    # Parse request body
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({
            "success": False,
            "error":   "Request body must be valid JSON with Content-Type: application/json"
        }), 400

    user_message = data.get("message", "").strip()
    use_hybrid   = data.get("use_hybrid", True)

    if not user_message:
        return jsonify({
            "success": False,
            "error":   "Field 'message' is required and cannot be empty."
        }), 400

    # Run chatbot
    result = chatbot.chat(user_message, use_hybrid=use_hybrid)

    return jsonify({
        "success": True,
        **result
    })


@app.route("/intents", methods=["GET"])
def list_intents():
    """
    Returns all available intents — useful for admin debugging.
    """
    return jsonify({
        "intents": list(chatbot.label2id.keys())
    })


# ─────────────────────────────────────────────────────────────
# DEMO — Test chatbot from command line (without Flask)
# ─────────────────────────────────────────────────────────────

def demo_chat():
    """
    Interactive demo — type queries and see responses.
    Run this function directly for quick testing.
    """
    print("\n" + "="*55)
    print(" SAMS Chatbot Demo ")
    print("="*55)
    print("Type your question. Type 'quit' to exit.\n")

    test_queries = [
        "How do I pay my rent?",
        "Can I have guests in my room?",
        "What happened to my application?",
        "Are pets allowed?",
        "Hello there",
        "I need asdfgh blahblah",   # should trigger fallback
    ]

    for query in test_queries:
        result = chatbot.chat(query)
        print(f"Student: {query}")
        print(f"Intent:  {result['intent']} (confidence: {result['confidence']}) via {result['method']}")
        print(f"SAMS:    {result['response']}")
        print("-" * 50)


if __name__ == "__main__":
    # ── Option A: Run the Flask API ───────────────────────────
    # Accessible at: http://localhost:5000/chat
    print("Starting SAMS Chatbot Flask API on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)

    # ── Option B: Run the demo (comment out app.run above) ───
    # demo_chat()
