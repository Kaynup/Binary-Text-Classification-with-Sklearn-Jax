import re
import numpy as np
import sentencepiece as spm
from nltk.tokenize import TweetTokenizer
from collections import Counter


def evaluate_tokenizer(tokenizer_func, texts, vocab=None, vocab_limit=None):
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
    
    avg_tokens = np.mean([len(toks) for toks in tokenized])
    median_tokens = np.median([len(toks) for toks in tokenized])
    
    return {
        "avg_tokens": avg_tokens,
        "median_tokens": median_tokens,
        "vocab_size": len(vocab),
        "oov_rate": oov_rate,
        "top_tokens": token_counts.most_common(10),
    }


def custom_tokenizer(text):
    text = text.lower()
    text = re.sub(r"http\S+", "<URL>", text)
    text = re.sub(r"@\w+", "<USER>", text)
    tokens = re.findall(
        r"[#@]?\w+|[:;=xX8][-^']?[)DPOp3]+|[^\w\s]",
        text, flags=re.UNICODE
    )
    return tokens


def nltk_tokenizer(text):
    nltk_tweet_tknzr = TweetTokenizer(preserve_case=False, strip_handles=False, reduce_len=False)
    return nltk_tweet_tknzr.tokenize(text)


def sentencepiece(corpus, vocab_size=8000, model_prefix="tweet_bpe"):
    with open("tmp_corpus.txt", "w", encoding="utf-8") as f:
        for t in corpus:
            f.write(t + "\n")
    spm.SentencePieceTrainer.train(
        input="tmp_corpus.txt", model_prefix=model_prefix,
        vocab_size=vocab_size, character_coverage=1.0, model_type="bpe"
    )
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    return sp