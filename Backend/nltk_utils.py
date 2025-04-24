import nltk
import numpy as np

# Tokenizer
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# Stemmer (optional)
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def stem(word):
    return stemmer.stem(word.lower())

# Build vocabulary
def build_vocab(all_sentences):
    vocab = set()
    for sentence in all_sentences:
        tokens = tokenize(sentence)
        vocab.update([stem(word) for word in tokens])
    vocab = sorted(vocab)
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    return vocab, word2idx

# Convert sentence → index tensor
def sentence_to_indices(sentence, word2idx, max_len):
    tokens = tokenize(sentence)
    stemmed = [stem(w) for w in tokens]
    indices = [word2idx[word] for word in stemmed if word in word2idx]
    
    # Pad or truncate
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    else:
        indices = indices[:max_len]

    return np.array(indices)
