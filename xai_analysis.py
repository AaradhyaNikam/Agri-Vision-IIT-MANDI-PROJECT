import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from lime import lime_image
from skimage.segmentation import mark_boundaries

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """
    Loads an image and preprocesses it for MobileNetV2.
    
    Args:
        img_path (str): Path to the target image.
        target_size (tuple): Expected input size of the model.
        
    Returns:
        preprocessed_img (np.ndarray): Tensor of shape (1, 224, 224, 3) ready for model prediction.
        raw_img (np.ndarray): Original image array in [0, 255] range for visualization.
    """
    # 1. Load image with target size matching MobileNetV2 expectations
    img = load_img(img_path, target_size=target_size)
    
    # 2. Convert to numpy array [0, 255]
    raw_img = img_to_array(img)
    
    # 3. Expand dimensions to create a batch of 1 (1, 224, 224, 3)
    img_array = np.expand_dims(np.copy(raw_img), axis=0)
    
    # 4. Apply MobileNetV2 specific preprocessing
    # MobileNetV2 expects pixel values in the range [-1, 1], so we use its native preprocess_input
    preprocessed_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    return preprocessed_img, raw_img


def get_nested_gradcam_heatmap(model, img_array, inner_model_name='mobilenetv2_1.00_224', pred_index=None):
    """
    Generates a Grad-CAM heatmap by splitting the nested architecture and using manual Auto-Diff.
    """
    # 1. Extract the base feature extractor (MobileNetV2)
    base_model = model.get_layer(inner_model_name)
    
    # 2. Reconstruct the Classifier Head using the exact trained layers from the summary
    classifier_input = tf.keras.Input(shape=base_model.output.shape[1:])
    x = classifier_input
    
    # These layer names match your terminal summary exactly
    head_layers = ['global_average_pooling2d', 'dropout', 'dense', 'dropout_1', 'dense_1']
    for layer_name in head_layers:
        x = model.get_layer(layer_name)(x)
        
    classifier_model = tf.keras.Model(classifier_input, x)
    
    # Convert numpy array to tensor
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    
    # 3. Manual Gradient Tracking
    with tf.GradientTape() as tape:
        # Forward pass through the base model
        last_conv_layer_output = base_model(img_tensor)
        
        # Manually force the tape to watch this intermediate feature map tensor
        tape.watch(last_conv_layer_output)
        
        # Forward pass the feature map through the classifier head
        preds = classifier_model(last_conv_layer_output)
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
            
        class_channel = preds[:, pred_index]

    # 4. Compute Gradients & Pool
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # 5. Weight the feature maps
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. Apply ReLU and Normalize
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()


def overlay_gradcam(original_img_array, heatmap, alpha=0.6):
    """
    Overlays the Grad-CAM heatmap onto the original image.
    
    Args:
        original_img_array (np.ndarray): Original image [0, 255].
        heatmap (np.ndarray): 2D heatmap [0, 1].
        alpha (float): Opacity of the heatmap overlay.
        
    Returns:
        np.ndarray: Superimposed image [0, 255].
    """
    # Scale heatmap to [0, 255]
    heatmap = np.uint8(255 * heatmap)
    
    # Use jet colormap to colorize heatmap
    try:
        jet = matplotlib.colormaps['jet']
    except AttributeError:
        # Fallback for older matplotlib versions
        import matplotlib.cm as cm
        jet = cm.get_cmap("jet")
        
    # Extract RGB values of the colormap (ignoring alpha channel)
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    # Resize the heatmap to match the original image dimensions
    jet_heatmap = tf.image.resize(jet_heatmap, (original_img_array.shape[0], original_img_array.shape[1]))
    jet_heatmap = jet_heatmap.numpy()
    
    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 255 * alpha + original_img_array
    
    # Clip values to valid image range and convert to uint8
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    
    return superimposed_img


def get_lime_explanation(model, raw_img, top_labels=1, num_samples=1000, num_features=5):
    """
    Generates a LIME (Local Interpretable Model-agnostic Explanations) superpixel explanation.
    
    LIME works by perturbing the input image (hiding different superpixels/textures) 
    and observing how the model's predictions change. It then fits a local linear model 
    to explain this behavior and map the most important textures.
    
    Args:
        model: The trained Keras model.
        raw_img (np.ndarray): The original image array [0, 255].
        top_labels (int): Number of top classes to explain.
        num_samples (int): Number of perturbations for LIME to generate.
        num_features (int): Number of most important superpixels to highlight.
        
    Returns:
        np.ndarray: Image showing LIME boundaries [0, 1].
    """
    explainer = lime_image.LimeImageExplainer()
    
    # Wrapper function for LIME: LIME generates perturbed images matching the scale 
    # of the input we give it. Since we pass raw_img [0, 255], we must apply 
    # MobileNetV2 preprocessing inside this wrapper before passing it to the model.
    def predict_function(images):
        # Ensure float32 and apply MobileNetV2 preprocessing (scales [0, 255] to [-1, 1])
        preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(np.copy(images).astype('float32'))
        return model.predict(preprocessed, verbose=0)

    # Generate the explanation (this takes some time depending on num_samples)
    explanation = explainer.explain_instance(
        raw_img.astype('double'), 
        predict_function, 
        top_labels=top_labels, 
        hide_color=0, 
        num_samples=num_samples
    )
    
    # Extract the explanation for the model's top predicted class
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], 
        positive_only=True,     # Highlight only areas that contributed positively to the class
        num_features=num_features,
        hide_rest=False         # Set to True if you want to black out everything except important regions
    )
    
    # Mark the boundaries of the important superpixels on the image
    # Note: mark_boundaries expects the image to be in [0, 1] range
    lime_img = mark_boundaries(temp / 255.0, mask)
    
    return lime_img


def plot_explanations(raw_img, gradcam_img, lime_img, save_path=None):
    """
    Plots the Original, Grad-CAM, and LIME images side-by-side.
    """
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Original Image
    ax[0].imshow(raw_img.astype('uint8'))
    ax[0].set_title('Original Leaf Image', fontsize=14, fontweight='bold')
    ax[0].axis('off')
    
    # 2. Grad-CAM
    ax[1].imshow(gradcam_img)
    ax[1].set_title('Grad-CAM Global Heatmap', fontsize=14, fontweight='bold')
    ax[1].axis('off')
    
    # 3. LIME
    ax[2].imshow(lime_img)
    ax[2].set_title('LIME Localized Superpixels', fontsize=14, fontweight='bold')
    ax[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Visualization saved to {save_path}")
        
    plt.show()


if __name__ == "__main__":
    # ---------------------------------------------------------
    # Example Execution Flow for Jaaniv Krushi Project
    # ---------------------------------------------------------
    
    # Paths (Modify these for your specific environment)
    MODEL_PATH = 'models/model.keras'
    TEST_IMAGE_PATH = 'test_leaf_image.png' # Replace with your test image path
    
    try:
        if not os.path.exists(TEST_IMAGE_PATH):
            print(f"Test image not found at '{TEST_IMAGE_PATH}'. Please update the path.")
        else:
            print("Loading Model...")
            # Note: If you encounter custom object errors, use custom_objects={'CustomLayer': CustomLayer}
            crop_model = tf.keras.models.load_model(MODEL_PATH)
            
            print("Loading and Preprocessing Image...")
            preprocessed_img, raw_img = load_and_preprocess_image(TEST_IMAGE_PATH)
            
            print("Computing Grad-CAM Heatmap...")
            heatmap = get_nested_gradcam_heatmap(
                model=crop_model, 
                img_array=preprocessed_img, 
                inner_model_name='mobilenetv2_1.00_224'
            )
            gradcam_display = overlay_gradcam(raw_img, heatmap)
            
            print("Generating LIME Explanation (this may take a moment)...")
            lime_display = get_lime_explanation(
                model=crop_model, 
                raw_img=raw_img, 
                num_samples=1000 # Adjust for speed vs. quality
            )
            
            print("Plotting Results...")
            plot_explanations(raw_img, gradcam_display, lime_display, save_path='xai_results.png')
            
    except Exception as e:
        print(f"An error occurred: {e}")
