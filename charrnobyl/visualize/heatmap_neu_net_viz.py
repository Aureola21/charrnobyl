
import matplotlib.pyplot as plt
import seaborn as sns
import torch

sns.set(style="whitegrid")

def plot_bigram_heatmap(model):
    '''Visualize the learned bigram probabilities as a heatmap.'''
    chars = list(model.s_to_i.keys())
    vocab_size = model.no_distinct_chars

    # get prob matrix
    P = torch.zeros((vocab_size, vocab_size))
    for i in range(vocab_size):
        x_enc = torch.nn.functional.one_hot(torch.tensor([i]), num_classes=vocab_size).float()
        logits = x_enc @ model.W
        counts = logits.exp()
        P[i] = counts / counts.sum(1, keepdim=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(P.detach().numpy(), xticklabels=chars, yticklabels=chars, cmap="Blues", cbar=True, square=True)
    plt.title("Bigram Probability Heatmap")
    plt.xlabel("Next character")
    plt.ylabel("Current character")
    plt.tight_layout()
    plt.show()