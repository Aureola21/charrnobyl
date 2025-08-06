
import matplotlib.pyplot as plt
import seaborn as sns


def plot_char_frequency(data):
    '''Visualize character frequency in the dataset.'''
    from collections import Counter
    all_chars = ''.join(data)
    counts = Counter(all_chars)
    chars, freqs = zip(*sorted(counts.items()))

    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(chars), y=list(freqs), palette='viridis')
    plt.title("Character Frequency in Training Data")
    plt.xlabel("Character")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()