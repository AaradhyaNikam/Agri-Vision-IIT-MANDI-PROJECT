import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def preprocess_image(image, label):
    """
    Applies MobileNetV2 preprocessing.
    """
    # Cast to float32 and apply standard MobileNetV2 preprocessing [-1, 1]
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

def main():
    print("Initializing Local PlantVillage Dataset...")
    
    dataset_dir = 'plantvillage dataset/color/'
    
    if not os.path.exists(dataset_dir):
        print(f"ERROR: Dataset directory not found at '{dataset_dir}'.")
        return

    # Load from local directory
    # shuffle=False is crucial so that true labels perfectly match predicted labels sequentially
    dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        batch_size=32,
        image_size=(224, 224),
        shuffle=False
    )
    
    # Get dynamic class names directly from the dataset object
    class_names = dataset.class_names
    num_classes = len(class_names)
    print(f"Successfully loaded dataset metadata. Found {num_classes} classes.")
    
    # Preprocess dataset
    print("Mapping preprocessing pipeline...")
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    # Load Model
    model_path = 'models/model.keras'
    print(f"Loading trained model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at '{model_path}'. Please ensure the path is correct.")
        return
        
    model = tf.keras.models.load_model(model_path)
    
    # Benchmarking
    print("Running predictions on the dataset (this may take a few moments)...")
    true_labels = []
    predicted_labels = []
    
    # Iterate through the batched dataset
    for step, (images, labels) in enumerate(dataset):
        preds = model.predict(images, verbose=0)
        pred_classes = np.argmax(preds, axis=1)
        
        true_labels.extend(labels.numpy())
        predicted_labels.extend(pred_classes)
        print(f"Processed batch {step + 1}... ({len(true_labels)} samples done)", end='\r')
    
    print(f"\nCompleted predictions for {len(true_labels)} samples.")
    
    # Create images output directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    # Classification Report
    print("Generating Classification Report...")
    report = classification_report(true_labels, predicted_labels, target_names=class_names, zero_division=0)
    
    report_path = 'images/classification_report.txt'
    with open(report_path, 'w') as f:
        f.write("Classification Report - Jaaniv Krushi Model (MobileNetV2)\n")
        f.write("="*60 + "\n")
        f.write(report)
        
    print(f"-> Classification report successfully saved to '{report_path}'.")
    
    # Confusion Matrix
    print("Generating High-Resolution Confusion Matrix plot...")
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Set up the matplotlib figure (Large size to accommodate 38 classes)
    plt.figure(figsize=(24, 20))
    
    # Use seaborn to draw the heatmap
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix - PlantVillage 38 Classes', fontsize=24, pad=20)
    plt.ylabel('True Label', fontsize=18)
    plt.xlabel('Predicted Label', fontsize=18)
    
    # Rotate x and y labels so they don't overlap
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    cm_path = 'images/confusion_matrix.png'
    plt.savefig(cm_path, dpi=300)
    print(f"-> Confusion matrix plot successfully saved to '{cm_path}'.")
    
    print("Benchmarking module completed successfully!")

if __name__ == '__main__':
    main()
