import math
import re
from collections import Counter, defaultdict


CORPUS = [
    "Artificial intelligence helps computers solve complex problems.",
    "Machine learning models learn patterns from data.",
    "Natural language processing works with text and speech.",
    "Language models estimate probabilities of word sequences.",
    "Tokenization splits text into separate words or tokens.",
    "Stop words are removed during text preprocessing.",
    "N gram models use local word context for prediction.",
    "Perplexity measures how well a language model predicts text.",
    "Text preprocessing often includes cleaning and normalization.",
    "Data driven systems improve when the training corpus grows.",
]

STOP_WORDS = {
    "a",
    "and",
    "are",
    "for",
    "from",
    "how",
    "into",
    "of",
    "or",
    "the",
    "with",
    "when",
    "well",
    "during",
}


def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


def preprocess_corpus(corpus):
    processed = []
    for text in corpus:
        tokens = tokenize(text)
        filtered = [token for token in tokens if token not in STOP_WORDS]
        processed.append(filtered)
    return processed


def generate_ngrams(tokens, n):
    padded = ["<s>"] * (n - 1) + tokens + ["</s>"]
    return [tuple(padded[index:index + n]) for index in range(len(padded) - n + 1)]


class NGramLanguageModel:
    def __init__(self, n, sentences):
        self.n = n
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.vocabulary = set()

        for sentence in sentences:
            self.vocabulary.update(sentence)
            for ngram in generate_ngrams(sentence, n):
                context = ngram[:-1]
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1

        self.vocabulary.update({"<s>", "</s>"})
        self.vocabulary_size = len(self.vocabulary)

    def probability(self, ngram):
        context = ngram[:-1]
        numerator = self.ngram_counts[ngram] + 1
        denominator = self.context_counts[context] + self.vocabulary_size
        return numerator / denominator

    def sentence_log_probability(self, sentence):
        total = 0.0
        for ngram in generate_ngrams(sentence, self.n):
            total += math.log(self.probability(ngram))
        return total

    def perplexity(self, sentences):
        token_count = 0
        log_probability = 0.0

        for sentence in sentences:
            token_count += len(sentence) + 1
            log_probability += self.sentence_log_probability(sentence)

        return math.exp(-log_probability / token_count)

    def top_predictions(self, context, limit=5):
        scores = []
        for word in sorted(self.vocabulary):
            if word == "<s>":
                continue
            ngram = tuple(context) + (word,)
            scores.append((word, self.probability(ngram)))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:limit]


def word_frequency(sentences):
    counter = Counter()
    for sentence in sentences:
        counter.update(sentence)
    return counter


def main():
    processed_corpus = preprocess_corpus(CORPUS)
    model = NGramLanguageModel(n=2, sentences=processed_corpus)
    frequencies = word_frequency(processed_corpus)

    print("Original corpus:")
    for text in CORPUS:
        print("-", text)

    print("\nProcessed corpus:")
    for sentence in processed_corpus:
        print(sentence)

    print("\nTop 10 word frequencies:")
    for word, count in frequencies.most_common(10):
        print(f"{word}: {count}")

    print("\nSample bigrams:")
    sample_bigrams = list(model.ngram_counts.items())[:10]
    for ngram, count in sample_bigrams:
        print(f"{ngram}: {count}")

    print(f"\nModel perplexity on the corpus: {model.perplexity(processed_corpus):.4f}")

    context = ("language",)
    print(f"\nTop predictions for context {context}:")
    for word, probability in model.top_predictions(context):
        print(f"{word}: {probability:.4f}")


if __name__ == "__main__":
    main()
