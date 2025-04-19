# This code was produced with the help of Shradhdha Bhalodia's code
# at https://www.kaggle.com/code/praveenraj001/deep-fake-image-classification-using-cnn
# The dataset used in this code is provided on the Kaggle website: 
# https://www.kaggle.com/datasets/saurabhbagchi/deepfake-image-detection/data

# Import Modules
import os,glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D
# from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load Data
# Resize original file and save to numpy array file
def load_data():
    # Define dataset path
    dataset_path = "/kaggle/input/deepfake-image-detection/test-20250112T065939Z-001/test"  # Update with your dataset path
    # List categories (fake & real)
    categories = [category for category in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, category))]
    
    # Load sample images (filter out directories)
    sample_images = []
    for category in categories:
        class_path = os.path.join(dataset_path, category)
        images = [img for img in os.listdir(class_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]  # Filter image files
        for img in images[:5]:  # Limit to first 5 images per category
            img_path = os.path.join(class_path, img)
            sample_images.append((img_path, category))

    # Display sample images
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, (img_path, label) in enumerate(sample_images):
        img = load_img(img_path)
        ax = axes[i // 5, i % 5]
        ax.imshow(img)
        ax.set_title(label)
        ax.axis("off")
    plt.show()
    return model
    exit()

def CNN_learning():
    # CNN Learning
    IMG_SIZE = (128, 128)  # Resize images
    BATCH_SIZE = 32

    # Function to load and preprocess images
    def load_and_preprocess_image(img_path):
        img = load_img(img_path, target_size=IMG_SIZE)  # Resize
        img_array = img_to_array(img) / 255.0  # Normalize
        return img_array
        exit()

    # Load all images and labels
    image_data = []
    labels = []
    label_map = {category: idx for idx, category in enumerate(categories)}
    for category in categories:
        class_path = os.path.join(dataset_path, category)
        images = os.listdir(class_path)
        for img in images:
            img_path = os.path.join(class_path, img)
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
    # Display augmented images
    fig, axes = plt.subplots(1, 5, figsize=(12, 4))
    for i, img in enumerate(augmented_images):  # Only one batch of 5 images
        if i >= 1:  # Only process the first batch
            break
        for j in range(5):  # Loop through the 5 images in the batch
            axes[j].imshow(img[j])  # img[j] is the individual image in the batch
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
        Dense(len(categories), activation="softmax")  # Number of categories
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    history = model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                    validation_data=(X_val, y_val),
                    epochs=50,
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
    model.save('/kaggle/working/fake_image_model.h5')

    exit()

def raw_predict(model):
    # Load and preprocess your image (adjust path and image size as necessary)
    input_img = load_img('/kaggle/input/fake-images/fake_129.jpg', target_size=(128, 128))  # Adjust to your image path
    input_array = img_to_array(input_img)
    input_array = np.expand_dims(input_array, axis=0)  # Add batch dimension

    # from tensorflow.keras.preprocessing.image import load_img, img_to_array

    # input_img = load_img('/kaggle/input/deepfake-image-detection/test-20250112T065939Z-001/test/fake/123.jpg', target_size=(128, 128))
    # input_array = img_to_array(input_img)


    # Normalize image to match the input used for training (if required)
    input_array = input_array / 255.0  # Assuming model was trained with normalized images

    # Get predictions from the model
    predictions = model(input_array)

    # Print the raw output (prediction probabilities for each class)
    print("Raw predictions (probabilities for each class):", predictions)

    # If it's a classification task, get the class with the highest probability
    predicted_class_idx = np.argmax(predictions, axis=-1)  # Get index of max probability
    print(f"Predicted class index: {predicted_class_idx}")

    # If you have class labels, you can map this index back to a class name
    class_labels = ['fake', 'real']  
    predicted_class_name = class_labels[predicted_class_idx[0]]  # For single image
    print(f"Predicted class name: {predicted_class_name}")

    # Optionally, if you have true labels, you can also print accuracy and compare
    # true_label = ...  # Load true label for comparison
    # accuracy = np.mean(predicted_class_idx == true_label)
    # print(f"Accuracy: {accuracy * 100:.2f}%")

    exit()

model = load_data()
CNN_learning(model)
raw_predict(model)
