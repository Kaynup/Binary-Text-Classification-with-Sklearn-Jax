"""
Tokenization utilities for Binary Text Sentiment Classification.
Includes regex-based tweet tokenizer, evaluation tools, and optional NLTK/SentencePiece adapters.
"""
import re
import numpy as np
from collections import Counter


def custom_tokenizer(text: str):
    """
    Regex-based tokenizer tailored for social media / sentiment text.
    Handles URLs, user mentions, emoticons, hashtags, and punctuation.
    """
    text = text.lower()
    text = re.sub(r"http\S+", "<URL>", text)
    text = re.sub(r"@\w+", "<USER>", text)
    tokens = re.findall(
        r"[#@]?\w+|[:;=xX8][-^']?[)DPOp3]+|[^\w\s]",
        text, flags=re.UNICODE
    )
    return tokens


def evaluate_tokenizer(tokenizer_func, texts, vocab=None, vocab_limit=None):
    """
    Evaluate vocabulary coverage, average tokens per sample, and out-of-vocabulary (OOV) rate.
    """
    tokenized = [tokenizer_func(t) for t in texts]
    all_tokens = [tok for sent in tokenized for tok in sent]
    token_counts = Counter(all_tokens)

    if vocab is None:
        if vocab_limit:
            vocab = {tok for tok, _ in token_counts.most_common(vocab_limit)}
        else:
            vocab = set(token_counts.keys())

    total_tokens = sum(len(toks) for toks in tokenized)
    oov_tokens = sum(1 for tok in all_tokens if tok not in vocab)
    oov_rate = oov_tokens / total_tokens if total_tokens > 0 else 0

    avg_tokens = float(np.mean([len(toks) for toks in tokenized])) if tokenized else 0.0
    median_tokens = float(np.median([len(toks) for toks in tokenized])) if tokenized else 0.0

    return {
        "avg_tokens": round(avg_tokens, 2),
        "median_tokens": round(median_tokens, 2),
        "vocab_size": len(vocab),
        "oov_rate": round(oov_rate, 4),
        "top_tokens": token_counts.most_common(10),
    }


def nltk_tokenizer(text: str):
    """Optional NLTK TweetTokenizer wrapper."""
    try:
        from nltk.tokenize import TweetTokenizer
        tknzr = TweetTokenizer(preserve_case=False, strip_handles=False, reduce_len=False)
        return tknzr.tokenize(text)
    except ImportError:
        return custom_tokenizer(text)


def train_sentencepiece(corpus, vocab_size=8000, model_prefix="tweet_bpe"):
    """Optional SentencePiece trainer."""
    try:
        import sentencepiece as spm
        temp_file = "tmp_corpus.txt"
        with open(temp_file, "w", encoding="utf-8") as f:
            for t in corpus:
                f.write(t + "\n")
        spm.SentencePieceTrainer.train(
            input=temp_file, model_prefix=model_prefix,
            vocab_size=vocab_size, character_coverage=1.0, model_type="bpe"
        )
        sp = spm.SentencePieceProcessor()
        sp.load(f"{model_prefix}.model")
        return sp
    except ImportError as e:
        raise ImportError(f"sentencepiece is not installed: {e}")