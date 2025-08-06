from charrnobyl.models.neural_net import NeuralNetModel
from charrnobyl.fetch_data import input_data
from charrnobyl.visualize.heatmap_neu_net_viz import plot_bigram_heatmap
from charrnobyl.visualize.char_freq_viz import plot_char_frequency

def main():
    # Initialize model
    model = NeuralNetModel(data=input_data)

    # Create training data
    x_train, y_train = model.create_train_data()

    # Train the model
    model.train(x_train, y_train, epochs=300, lr=10.0, reg_lambda=0.01)

    # Evaluate the model
    print("\n🔍 Evaluating the trained model on given data:")
    model.evaluate(["sehaj"], print_data=True)

    # Sample from the model
    print("\n🎲 Sampling new words:")
    model.sample(num=10)

    #can also add viz script here
    viz = input("Do you want to visualize the heatmap of the neural network model? (y/n): ").strip().lower()
    if viz == 'y':
        plot_bigram_heatmap(model)
        print()

    #vizualize the character frequency
    viz_freq= input("Do you want to visualize character frequency? (y/n): ").strip().lower()
    if viz_freq == 'y':
        print("\n📊 Visualizing character frequency in the dataset:")
        plot_char_frequency(input_data)
        print()

    # Plot training loss
    print("\n📉 Plotting training loss over epochs:")
    model.plot_train_loss()


    print("\n✅ Model training and evaluation completed.")

if __name__ == "__main__":
    main()

