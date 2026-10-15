"""Feature extraction and vocabulary tokenization for microservice logs."""
import re
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from src.orca.preprocessing.normalizer import SERVICES, SERVICE2IDX, normalize_service_name

ERROR_KEYWORDS = {
    "error", "exception", "failed", "failure", "timeout", "timed", "fatal",
    "warn", "warning", "refused", "unavailable", "exhausted", "killed", "oom",
    "disconnect", "panic", "critical"
}


class LogTokenizer:
    """Lightweight character/word tokenizer for log messages."""

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.word2idx: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3}
        self.idx2word: Dict[int, str] = {0: "<PAD>", 1: "<UNK>", 2: "<CLS>", 3: "<SEP>"}
        self.is_fitted = False

    def fit(self, messages: List[str]) -> "LogTokenizer":
        """Build vocabulary from log messages."""
        freq: Dict[str, int] = {}
        for msg in messages:
            if not isinstance(msg, str):
                continue
            words = re.findall(r"[A-Za-z0-9_]+", msg.lower())
            for w in words:
                freq[w] = freq.get(w, 0) + 1

        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        for w, _ in sorted_words:
            if len(self.word2idx) >= self.vocab_size:
                break
            if w not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[w] = idx
                self.idx2word[idx] = w

        self.is_fitted = True
        return self

    def encode(self, text: str, max_length: int = 64) -> List[int]:
        """Convert log string to token ID sequence with padding/truncation."""
        if not isinstance(text, str):
            return [0] * max_length
        words = re.findall(r"[A-Za-z0-9_]+", text.lower())
        tokens = [2]  # <CLS>
        for w in words:
            if len(tokens) >= max_length - 1:
                break
            tokens.append(self.word2idx.get(w, 1))  # 1 is <UNK>
        tokens.append(3)  # <SEP>
        if len(tokens) < max_length:
            tokens.extend([0] * (max_length - len(tokens)))  # pad with 0
        return tokens[:max_length]


def extract_window_log_features(
    window_logs: pd.DataFrame,
    tokenizer: LogTokenizer,
    max_tokens: int = 64,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Extract log tokens and per-service log statistics for a time window.
    
    Returns:
        token_ids: np.ndarray of shape (max_tokens,)
        service_stats: np.ndarray of shape (20,) [10 counts, 10 error counts]
        aggregated_text: combined string of log events in window
    """
    num_services = len(SERVICES)
    counts = np.zeros(num_services, dtype=np.float32)
    error_counts = np.zeros(num_services, dtype=np.float32)

    if window_logs.empty:
        token_ids = np.zeros(max_tokens, dtype=np.int64)
        service_stats = np.zeros(num_services * 2, dtype=np.float32)
        return token_ids, service_stats, ""

    messages = []
    for _, row in window_logs.iterrows():
        svc = normalize_service_name(row.get("container_name"))
        msg = str(row.get("message", "")).lower()
        if svc in SERVICE2IDX:
            idx = SERVICE2IDX[svc]
            counts[idx] += 1.0
            if any(k in msg for k in ERROR_KEYWORDS):
                error_counts[idx] += 1.0
        messages.append(msg)

    # Aggregate text for tokenizer
    combined_text = " ; ".join(messages[:10])  # Sample top messages in window
    token_ids = np.array(tokenizer.encode(combined_text, max_length=max_tokens), dtype=np.int64)
    service_stats = np.concatenate([counts, error_counts]).astype(np.float32)

    return token_ids, service_stats, combined_text
