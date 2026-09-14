import random
import re
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import Counter

from tqdm import tqdm

def load_data(corpus_path, min_freq=20):
    corpus_text = ""
    with open(corpus_path, encoding="utf-8") as f:
        corpus_text = f.read()
        
    corpus_text = corpus_text.lower()
    corpus_text = re.sub(r'[^a-z\s]', ' ', corpus_text)
    corpus_text = re.sub(r'\s+', ' ', corpus_text).strip()


    tokens = corpus_text.split()
    word_counts = Counter(tokens)

    vocab = ['<unk>'] + [word for word, count in word_counts.items() if count >= min_freq]
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}, Total tokens: {len(tokens)}")

    word_to_ix = {word: i for i, word in enumerate(vocab)}
    ix_to_word = {i: word for i, word in enumerate(vocab)}
    unk_ix = word_to_ix['<unk>']
    tokens = [word if word in word_to_ix else '<unk>' for word in tokens]
    word_counts = Counter(tokens)

    return tokens, word_counts, word_to_ix, ix_to_word, unk_ix

def create_skipgram_data(tokens, window_size, word_to_ix):
    data = []
    progress_bar = tqdm(tokens, desc="Creating Skip-Gram Data")
    for i, center_word in enumerate(progress_bar):
        if center_word == '<unk>':
            continue
        for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
            if i == j:
                continue
            context_word = tokens[j]
            if context_word == '<unk>':
                continue
            data.append((word_to_ix[center_word], word_to_ix[context_word]))
    return data

class Word2Vec(nn.Module):
    # 1. Word2Vec model with separate center and outside embedding layers.
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

        # A small random initialization for center vectors and zero initialization
        # for outside vectors are commonly used for skip-gram Word2Vec.
        bound = 0.5 / embedding_dim
        nn.init.uniform_(self.in_embed.weight, -bound, bound)
        nn.init.zeros_(self.out_embed.weight)

    def forward(self, center_indices, outside_indices):
        center_vectors = self.in_embed(center_indices)
        outside_vectors = self.out_embed(outside_indices)

        # For K negative samples, broadcast the single center vector across the
        # K outside vectors. For a positive pair, both tensors are already 2-D.
        if outside_vectors.dim() == center_vectors.dim() + 1:
            center_vectors = center_vectors.unsqueeze(1)

        return (center_vectors * outside_vectors).sum(dim=-1)

class NegativeSampler:
    # 3. Draw non-context words from the smoothed unigram distribution.
    def __init__(self, word_counts, word_to_ix=None, power=0.75):
        # word_to_ix is optional so the sampler remains compatible with the
        # original one-argument constructor in the provided template.
        if word_to_ix is None:
            words = ['<unk>'] + [word for word in word_counts if word != '<unk>']
            word_to_ix = {word: i for i, word in enumerate(words)}

        self.word_to_ix = word_to_ix
        probabilities = torch.zeros(len(word_to_ix), dtype=torch.float)
        for word, count in word_counts.items():
            if word in word_to_ix:
                probabilities[word_to_ix[word]] = float(count) ** power

        # <unk> is not used as a positive training word, so do not sample it as
        # a negative word either.
        if '<unk>' in word_to_ix:
            probabilities[word_to_ix['<unk>']] = 0.0

        if probabilities.sum() == 0:
            raise ValueError("Cannot sample negatives from an empty vocabulary")
        self.probabilities = probabilities / probabilities.sum()

    def get_negative_samples(self, context_word_idx, k):
        if k < 0:
            raise ValueError("k must be non-negative")
        if not 0 <= context_word_idx < len(self.probabilities):
            raise IndexError("context_word_idx is outside the vocabulary")
        if k == 0:
            return torch.empty(0, dtype=torch.long)

        # The true context word must not appear among the negative samples.
        probabilities = self.probabilities.clone()
        probabilities[context_word_idx] = 0.0
        total_probability = probabilities.sum()
        if total_probability == 0:
            raise ValueError("No non-context word is available for negative sampling")

        probabilities /= total_probability
        return torch.multinomial(probabilities, k, replacement=True)

def train(model, training_data, optimizer, loss_fn, epochs, negative_sampler, k_negative_samples=5):
    device = next(model.parameters()).device
    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(training_data)
        progress_bar = tqdm(training_data, desc=f"Epoch {epoch + 1}/{epochs}")
        for i, (center_word_idx, context_word_idx) in enumerate(progress_bar):
            center_tensor = torch.tensor([center_word_idx], dtype=torch.long, device=device)
            context_tensor = torch.tensor([context_word_idx], dtype=torch.long, device=device)
            neg_sample_indices = negative_sampler.get_negative_samples(
                context_word_idx, k_negative_samples
            ).to(device)

            # 4. Optimize the positive pair and its K negative pairs together.
            optimizer.zero_grad()

            positive_scores = model(center_tensor, context_tensor)
            negative_scores = model(center_tensor, neg_sample_indices)

            positive_loss = loss_fn(positive_scores, torch.ones_like(positive_scores))
            negative_loss = loss_fn(negative_scores, torch.zeros_like(negative_scores))
            loss = positive_loss + negative_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()


            if i % 1000 == 0:
                progress_bar.set_postfix({'loss': f'{total_loss / (i + 1):.4f}'})

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(training_data):.4f}")

def visualize_embeddings(model, ix_to_word):
    embeddings = model.in_embed.weight.detach().cpu().numpy()
    pca = PCA(n_components=2)
    result = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    plt.scatter(result[:, 0], result[:, 1])
    
    for i, word in ix_to_word.items():
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
        
    plt.title("Word2Vec Embeddings with Negative Sampling")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.show()

if __name__== "__main__":
    print("PyTorch version:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    corpus_path = "corpus.txt"
    tokens, word_counts, word_to_ix, ix_to_word, unk_ix = load_data(f"{corpus_path}", min_freq=1)
    vocab = list(word_to_ix.keys())
    vocab_size = len(vocab)

    training_data = create_skipgram_data(tokens, window_size=5, word_to_ix=word_to_ix)
    print(f"Generated {len(training_data)} pairs.")
    print(f"Vocabulary size: {vocab_size}")

    EMBEDDING_DIM = 32
    LEARNING_RATE = 0.01
    EPOCHS = 5
    K_NEGATIVE_SAMPLES = 5

    model = Word2Vec(vocab_size, EMBEDDING_DIM)
    # 2. Summing over the K negative examples
    # matches Eq. (8) in the assignment.
    loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    negative_sampler = NegativeSampler(word_counts, word_to_ix)

    # train(model, training_data, optimizer, loss_fn, EPOCHS)
    train(model, training_data, optimizer, loss_fn, EPOCHS, negative_sampler, K_NEGATIVE_SAMPLES)
    print("Training complete!")

    visualize_embeddings(model, ix_to_word)
