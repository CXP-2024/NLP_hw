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
    # TODO: 1. Implement the Word2Vec model with two embedding layers
    pass

class NegativeSampler:
    # TODO: 3. Implement negative sampling
    pass

def train(model, training_data, optimizer, loss_fn, epochs, negative_sampler, k_negative_samples=5):
    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(training_data)
        progress_bar = tqdm(training_data, desc=f"Epoch {epoch + 1}/{epochs}")
        for i, (center_word_idx, context_word_idx) in enumerate(progress_bar):
            center_tensor = torch.tensor([center_word_idx], dtype=torch.long)
            neg_sample_indices = negative_sampler.get_negative_samples(context_word_idx, k_negative_samples)

            # TODO: 4. Train the model using negative sampling
            

            if i % 1000 == 0:
                progress_bar.set_postfix({'loss': f'{total_loss / (i + 1):.4f}'})

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(training_data):.4f}")

def visualize_embeddings(model, ix_to_word):
    embeddings = model.in_embed.weight.data.cpu().numpy()
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
    loss_fn = None # TODO: 2. Define the loss function
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    negative_sampler = NegativeSampler(word_counts)

    # train(model, training_data, optimizer, loss_fn, EPOCHS)
    train(model, training_data, optimizer, loss_fn, EPOCHS, negative_sampler, K_NEGATIVE_SAMPLES)
    print("Training complete!")

    visualize_embeddings(model, ix_to_word)
