from sklearn.model_selection import train_test_split

# Assume you have a dataset of video frames and corresponding labels
# X_train, X_test: Video frames, y_train, y_test: Corresponding labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.20, random_state=24)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=24)

y_train_encoded = to_categorical(y_train, num_classes=2)  # Assuming you have 2 classes
y_val_encoded = to_categorical(y_val, num_classes=2)

history = combined_model.fit(X_train, y_train_encoded, epochs=350, batch_size=6, validation_data=(X_val, y_val_encoded))
import matplotlib.pyplot as plt

def plot_history(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

# Assuming you have a history object returned by model.fit()
plot_history(history)
y_test_encoded = to_categorical(y_test, num_classes=2)
# Evaluate the model on the test set
test_loss, test_acc = combined_model.evaluate(X_test, y_test_encoded, batch_size=6)
print(f'Test accuracy: {test_acc}')
