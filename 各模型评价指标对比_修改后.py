import matplotlib.pyplot as plt

# Example accuracy values for the models (replace these with your actual values)
accuracy = {
    "LeNet": 98.93,
    "AlexNet": 98.90,
    "VGG": 98.96,
    "ResidualAlexNet-ReLU": 99.00,
    "ResidualAlexNet-PReLU": 98.77,
    "ResidualAlexNet-LReLU": 99.19
}

# Create the bar chart
model_names = list(accuracy.keys())
accuracy_values = list(accuracy.values())
colors = ["blue", "green", "red", "cyan", "magenta", "yellow"]

# Set the figure size
plt.figure(figsize=(12, 6))

plt.bar(model_names, accuracy_values, color=colors)
plt.xlabel("Models")
plt.ylabel("Classification Accuracy (%)")
plt.title("Comparison of Model Performance on MNIST Dataset")

# Set the y-axis limits
plt.ylim(98, 100)

# Rotate the x-axis labels to prevent overlap
plt.xticks(rotation=60)

# Adjust the layout to fit the labels properly
plt.tight_layout()

# Display the chart
plt.show()
