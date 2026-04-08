"""
Comprehensive NLP examples in Python.

This script covers core NLP topics and provides 2 examples for each topic.
It is designed as a learning/reference file and may require installing:
    pip install nltk spacy scikit-learn textblob gensim
and for spaCy model:
    python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable


# ================================
# 1) Text Cleaning & Normalization
# ================================
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def demo_text_cleaning() -> None:
    examples = [
        "NLP is AMAZING!!! #AI @2026",
        "Email me at hello@example.com -- ASAP.",
    ]
    print("\n[1] Text Cleaning & Normalization")
    for i, ex in enumerate(examples, 1):
        print(f"Example {i} Raw   : {ex}")
        print(f"Example {i} Clean : {clean_text(ex)}")


# ================================
# 2) Tokenization
# ================================
def word_tokenize_simple(text: str) -> list[str]:
    return clean_text(text).split()


def sentence_tokenize_simple(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def demo_tokenization() -> None:
    ex1 = "I love NLP. It is fun and useful!"
    ex2 = "Tokenize this sentence, please; into words."

    print("\n[2] Tokenization")
    print("Example 1 Sentence Tokens:", sentence_tokenize_simple(ex1))
    print("Example 2 Word Tokens    :", word_tokenize_simple(ex2))


# ================================
# 3) Stopword Removal
# ================================
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "am", "i", "me", "my", "we",
    "you", "he", "she", "it", "they", "and", "or", "but", "if", "to", "of", "in",
    "on", "for", "with", "this", "that", "these", "those", "as", "at", "by",
}


def remove_stopwords(tokens: Iterable[str]) -> list[str]:
    return [t for t in tokens if t not in STOPWORDS]


def demo_stopwords() -> None:
    ex1 = word_tokenize_simple("This is a simple example of stopword removal")
    ex2 = word_tokenize_simple("They are working on the project in the lab")

    print("\n[3] Stopword Removal")
    print("Example 1:", remove_stopwords(ex1))
    print("Example 2:", remove_stopwords(ex2))


# ================================
# 4) Stemming (rule-based, simple)
# ================================
def simple_stem(word: str) -> str:
    for suffix in ("ing", "edly", "edly", "ed", "ly", "s"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[: -len(suffix)]
    return word


def demo_stemming() -> None:
    ex1 = ["playing", "played", "plays", "player"]
    ex2 = ["happily", "walking", "walked", "walks"]

    print("\n[4] Stemming")
    print("Example 1:", [(w, simple_stem(w)) for w in ex1])
    print("Example 2:", [(w, simple_stem(w)) for w in ex2])


# ================================
# 5) Lemmatization (lightweight dictionary)
# ================================
LEMMA_MAP = {
    "mice": "mouse",
    "geese": "goose",
    "children": "child",
    "better": "good",
    "running": "run",
    "ate": "eat",
}


def simple_lemmatize(word: str) -> str:
    return LEMMA_MAP.get(word, word)


def demo_lemmatization() -> None:
    ex1 = ["mice", "children", "running"]
    ex2 = ["geese", "better", "ate"]

    print("\n[5] Lemmatization")
    print("Example 1:", [(w, simple_lemmatize(w)) for w in ex1])
    print("Example 2:", [(w, simple_lemmatize(w)) for w in ex2])


# ================================
# 6) N-grams
# ================================
def make_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def demo_ngrams() -> None:
    ex1 = word_tokenize_simple("natural language processing is fun")
    ex2 = word_tokenize_simple("machine learning powers nlp tasks")

    print("\n[6] N-grams")
    print("Example 1 Bigrams:", make_ngrams(ex1, 2))
    print("Example 2 Trigrams:", make_ngrams(ex2, 3))


# ================================
# 7) Bag of Words
# ================================
def bag_of_words(tokens: list[str]) -> dict[str, int]:
    return dict(Counter(tokens))


def demo_bow() -> None:
    ex1 = word_tokenize_simple("nlp nlp text processing")
    ex2 = word_tokenize_simple("data science and data analysis")

    print("\n[7] Bag of Words")
    print("Example 1:", bag_of_words(ex1))
    print("Example 2:", bag_of_words(ex2))


# ================================
# 8) TF-IDF (manual miniature)
# ================================
def tf(term: str, doc: list[str]) -> float:
    return doc.count(term) / len(doc) if doc else 0.0


def idf(term: str, corpus: list[list[str]]) -> float:
    import math

    docs_with_term = sum(1 for d in corpus if term in d)
    return math.log((1 + len(corpus)) / (1 + docs_with_term)) + 1


def tfidf(term: str, doc: list[str], corpus: list[list[str]]) -> float:
    return tf(term, doc) * idf(term, corpus)


def demo_tfidf() -> None:
    docs = [
        word_tokenize_simple("nlp is fun and useful"),
        word_tokenize_simple("nlp and machine learning"),
    ]
    print("\n[8] TF-IDF")
    print("Example 1 TF-IDF('nlp' in doc1):", round(tfidf("nlp", docs[0], docs), 4))
    print("Example 2 TF-IDF('useful' in doc1):", round(tfidf("useful", docs[0], docs), 4))


# ================================
# 9) Part-of-Speech Tagging (rule-based demo)
# ================================
def simple_pos_tag(tokens: list[str]) -> list[tuple[str, str]]:
    tags = []
    for t in tokens:
        if t.endswith("ing"):
            tag = "VBG"
        elif t.endswith("ed"):
            tag = "VBD"
        elif t.endswith("ly"):
            tag = "RB"
        elif t in {"a", "an", "the"}:
            tag = "DT"
        else:
            tag = "NN"
        tags.append((t, tag))
    return tags


def demo_pos() -> None:
    ex1 = word_tokenize_simple("the running dog barked loudly")
    ex2 = word_tokenize_simple("a trained model predicts quickly")

    print("\n[9] POS Tagging")
    print("Example 1:", simple_pos_tag(ex1))
    print("Example 2:", simple_pos_tag(ex2))


# ================================
# 10) Named Entity Recognition (pattern demo)
# ================================
def simple_ner(text: str) -> list[tuple[str, str]]:
    entities: list[tuple[str, str]] = []
    for token in text.split():
        if token.istitle():
            entities.append((token, "PROPN"))
        if token.endswith("Inc") or token.endswith("Corp"):
            entities.append((token, "ORG"))
    return entities


def demo_ner() -> None:
    ex1 = "Alice joined OpenAI Inc in SanFrancisco"
    ex2 = "Bob visited Microsoft Corp headquarters"

    print("\n[10] Named Entity Recognition")
    print("Example 1:", simple_ner(ex1))
    print("Example 2:", simple_ner(ex2))


# ================================
# 11) Sentiment Analysis (lexicon demo)
# ================================
POS_WORDS = {"good", "great", "awesome", "love", "happy", "excellent"}
NEG_WORDS = {"bad", "terrible", "hate", "awful", "sad", "poor"}


def simple_sentiment(text: str) -> str:
    tokens = word_tokenize_simple(text)
    score = sum(1 for t in tokens if t in POS_WORDS) - sum(1 for t in tokens if t in NEG_WORDS)
    if score > 0:
        return "Positive"
    if score < 0:
        return "Negative"
    return "Neutral"


def demo_sentiment() -> None:
    ex1 = "I love this awesome NLP project"
    ex2 = "This model is awful and bad"

    print("\n[11] Sentiment Analysis")
    print("Example 1:", simple_sentiment(ex1))
    print("Example 2:", simple_sentiment(ex2))


# ================================
# 12) Text Classification (keyword-based)
# ================================
def simple_classifier(text: str) -> str:
    tokens = set(word_tokenize_simple(text))
    if tokens & {"match", "team", "score", "tournament"}:
        return "Sports"
    if tokens & {"stock", "market", "investment", "bank"}:
        return "Finance"
    return "General"


def demo_classification() -> None:
    ex1 = "The team won the match with a high score"
    ex2 = "Stock market investment can be risky"

    print("\n[12] Text Classification")
    print("Example 1:", simple_classifier(ex1))
    print("Example 2:", simple_classifier(ex2))


# ================================
# 13) Topic Extraction (top keywords)
# ================================
def top_keywords(text: str, k: int = 3) -> list[tuple[str, int]]:
    tokens = remove_stopwords(word_tokenize_simple(text))
    return Counter(tokens).most_common(k)


def demo_topic_extraction() -> None:
    ex1 = "NLP models process language and language data for smart applications"
    ex2 = "Farming soil moisture crop yield crop health and crop monitoring"

    print("\n[13] Topic Extraction")
    print("Example 1:", top_keywords(ex1, 3))
    print("Example 2:", top_keywords(ex2, 3))


# ================================
# 14) Text Similarity (Jaccard)
# ================================
def jaccard_similarity(a: str, b: str) -> float:
    sa = set(word_tokenize_simple(a))
    sb = set(word_tokenize_simple(b))
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def demo_similarity() -> None:
    ex1_a = "nlp and machine learning"
    ex1_b = "machine learning for nlp"

    ex2_a = "crop recommendation using rainfall"
    ex2_b = "movie recommendation using ratings"

    print("\n[14] Text Similarity")
    print("Example 1:", round(jaccard_similarity(ex1_a, ex1_b), 4))
    print("Example 2:", round(jaccard_similarity(ex2_a, ex2_b), 4))


# ================================
# 15) Spell Correction (edit-distance-like lookup)
# ================================
DICTIONARY = {"language", "processing", "recommendation", "machine", "learning", "model"}


def simple_spell_correct(word: str) -> str:
    # Tiny nearest heuristic by length + prefix overlap.
    best = word
    best_score = -1
    for candidate in DICTIONARY:
        common_prefix = 0
        for a, b in zip(word, candidate):
            if a == b:
                common_prefix += 1
            else:
                break
        score = common_prefix - abs(len(word) - len(candidate))
        if score > best_score:
            best_score = score
            best = candidate
    return best


def demo_spell_correction() -> None:
    ex1 = "langauge"
    ex2 = "recomendation"

    print("\n[15] Spell Correction")
    print("Example 1:", ex1, "->", simple_spell_correct(ex1))
    print("Example 2:", ex2, "->", simple_spell_correct(ex2))


# ================================
# Runner
# ================================
def main() -> None:
    print("Comprehensive NLP Demo (2 examples per topic)")
    demo_text_cleaning()
    demo_tokenization()
    demo_stopwords()
    demo_stemming()
    demo_lemmatization()
    demo_ngrams()
    demo_bow()
    demo_tfidf()
    demo_pos()
    demo_ner()
    demo_sentiment()
    demo_classification()
    demo_topic_extraction()
    demo_similarity()
    demo_spell_correction()


if __name__ == "__main__":
    main()
