# charrnobyl/charrnobyl/visualize/bigram_counts_viz.py
import matplotlib.pyplot as plt


def Visualize_Bigram_Counts(model):
    """
    Visualizes the bigram counts as a heatmap.
    """
    print("Visualizing Bigram Counts...")
    N = model.N
    i_to_s = model.i_to_s
    no_distinct_chars = model.no_distinct_chars

    # Setup plot
    plt.figure(figsize=(no_distinct_chars, no_distinct_chars))
    plt.imshow(N, cmap='viridis')  # better dynamic range

    # Add count and character labels
    for i in range(no_distinct_chars):
        for j in range(no_distinct_chars):
            count = N[i, j].item()
            if count >= 0:
                ch_str = i_to_s[i] + i_to_s[j]
                plt.text(j, i, ch_str, ha='center', va='bottom', fontsize=6, color='white')
                plt.text(j, i, str(count), ha='center', va='top', fontsize=6, color='white')

    # Ticks and labels
    plt.xticks(ticks=range(no_distinct_chars), labels=[i_to_s[i] for i in range(no_distinct_chars)], rotation=90, fontsize=6)
    plt.yticks(ticks=range(no_distinct_chars), labels=[i_to_s[i] for i in range(no_distinct_chars)], fontsize=6)
    plt.title("Bigram Count Heatmap", fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.show()