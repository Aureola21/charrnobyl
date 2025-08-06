from charrnobyl.models.bigram import BigramModel
from charrnobyl.fetch_data import input_data
from charrnobyl.visualize.bigram_counts_viz import Visualize_Bigram_Counts
from charrnobyl.visualize.char_freq_viz import plot_char_frequency

def main():
    # Load input data
    model = BigramModel(input_data)

    # Build count matrix
    model.build_bigram()
    print()

    num= int(input("How many words do you want to sample? (default=10): ") or 10)
    # Sample new words
    print("🔤 Sampled Words:")
    model.sample(num=num)
    print()
    # Visualize
    viz= input("Do you want to visualize the bigram counts? (y/n): ").strip().lower()
    if viz=='y':
        print("Visualizing Bigram Counts...")
        Visualize_Bigram_Counts(model)
    print()
    #vizualize the character frequency
    viz_freq= input("Do you want to visualize character frequency? (y/n): ").strip().lower()
    if viz_freq == 'y':
        print("\n📊 Visualizing character frequency in the dataset:")
        plot_char_frequency(input_data)
        print()

    # Evaluate the model
    print("Evaluating the model...")
    model.evaluate()


if __name__ == "__main__":
    main()