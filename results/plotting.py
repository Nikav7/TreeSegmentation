import pandas as pd
import matplotlib.pyplot as plt

# load history
history_path = "results/history_model5.csv"
history = pd.read_csv(history_path)

# training and validation accuracies
plt.figure(figsize=(10, 5))
plt.plot(history["accuracy"], label="Training Accuracy", color="blue")
plt.plot(history["val_accuracy"], label="Validation Accuracy", color="orange")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy Over Epochs")
plt.legend()
plt.grid()
plt.show()

# training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(history["loss"], label="Training Loss", color="blue")
plt.plot(history["val_loss"], label="Validation Loss", color="orange")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.grid()
plt.show()
