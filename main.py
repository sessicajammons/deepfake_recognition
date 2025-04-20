# This code was adapted with the help of Shradhdha Bhalodia's code
# at https://www.kaggle.com/code/praveenraj001/deep-fake-image-classification-using-cnn
# The dataset used in this code is provided on the Kaggle website: 
# https://www.kaggle.com/datasets/saurabhbagchi/deepfake-image-detection/data

# Import Modules
import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load Data
def load_data(dataset_path):
    """
    Load image paths and categories from the dataset, display sample images.
    Args:
        dataset_path (str): Path to the dataset folder containing category subfolders.
    Returns:
        tuple: (img_paths, categories) where img_paths is a list of image file paths
               and categories is a list of category names.
    """
    # List categories (fake & real)
    categories = [category for category in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, category))]

    # Collect all image paths and sample images for display
    img_paths = []
    sample_images = []
    for category in categories:
        class_path = os.path.join(dataset_path, category)
        images = [img for img in os.listdir(class_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img in images:
            img_path = os.path.join(class_path, img)
            img_paths.append((img_path, category))  # Store path and category
            if len(sample_images) < 10 and len([s for s, c in sample_images if c == category]) < 5:
                sample_images.append((img_path, category))  # Limit to 5 per category

    # Display sample images
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, (img_path, label) in enumerate(sample_images):
        img = load_img(img_path)
        ax = axes[i // 5, i % 5]
        ax.imshow(img)
        ax.set_title(label)
        ax.axis("off")
    plt.show()

    return img_paths, categories

def CNN_learning(img_paths, categories):
    """
    Train a CNN model using the provided image paths and categories.
    Args:
        img_paths (list): List of tuples (image_path, category).
        categories (list): List of category names.
    Returns:
        model: Trained Keras model.
    """
    IMG_SIZE = (128, 128)
    BATCH_SIZE = 32

    # Function to load and preprocess images
    def load_and_preprocess_image(img_path):
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        return img_array

    # Load all images and labels
    image_data = []
    labels = []
    label_map = {category: idx for idx, category in enumerate(categories)}
    for img_path, category in img_paths:
        image_data.append(load_and_preprocess_image(img_path))
        labels.append(label_map[category])

    # Convert to numpy arrays
    image_data = np.array(image_data)
    labels = np.array(labels)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")

    # Define augmentation parameters
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    # Generate augmented images
    augmented_images = datagen.flow(X_train[:5], batch_size=5)
    fig, axes = plt.subplots(1, 5, figsize=(12, 4))
    for i, img in enumerate(augmented_images):
        if i >= 1:
            break
        for j in range(5):
            axes[j].imshow(img[j])
            axes[j].axis("off")
    plt.show()

    # Build CNN Model
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(len(categories), activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    history = model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                        validation_data=(X_val, y_val),
                        epochs=10,
                        verbose=1)

    # Plot Accuracy and Loss Curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.title("Accuracy")
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.title("Loss")
    plt.show()

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save the trained model
    model.save('./fake_image_model.h5')

    return model

def raw_predict(model, test_img_path):
    """
    Predict the class of a single test image.
    Args:
        model: Trained Keras model.
        test_img_path (str): Path to the test image.
    """
    input_img = load_img(test_img_path, target_size=(128, 128))
    input_array = img_to_array(input_img)
    input_array = np.expand_dims(input_array, axis=0)
    input_array = input_array / 255.0

    predictions = model.predict(input_array)
    print("Raw predictions (probabilities for each class):", predictions)

    predicted_class_idx = np.argmax(predictions, axis=-1)
    print(f"Predicted class index: {predicted_class_idx}")

    class_labels = ['fake', 'real']
    predicted_class_name = class_labels[predicted_class_idx[0]]
    print(f"Predicted class name: {predicted_class_name}")

# Main execution
# Set dataset path (update based on environment)
# Download latest version
dataset_path = kagglehub.dataset_download("saurabhbagchi/deepfake-image-detection")
print("Path to dataset files:", dataset_path)
# If running locally, update to where the dataset is extracted, e.g.:
# dataset_path = "./new_dataset/test"

img_paths, categories = load_data(dataset_path)
model = CNN_learning(img_paths, categories)

# Example test image path (update to a valid image from your dataset)
test_img_path = "/kaggle/input/deepfake-image-detection/test-20250112T065939Z-001/test/fake/123.jpg"
raw_predict(model, test_img_path)

