import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import keras

# ================== 1. ENABLE SAFE MODEL LOADING ==================
keras.config.enable_unsafe_deserialization()

# ================== 2. LOAD TRAINED MODELS ==================
# Paths to trained models
generator_path = r"C:\Gen AI project\GAN training\models_2\generator_rice1_latest.keras"
discriminator_path = r"C:\Gen AI project\GAN training\models_2\discriminator_rice1_latest.keras"

# Load trained models (Discriminator not needed for inference but loaded for verification)
generator = tf.keras.models.load_model(generator_path, compile=False, safe_mode=False)
discriminator = tf.keras.models.load_model(discriminator_path, compile=False, safe_mode=False)

print("✅ RICE1 Generator and Discriminator models loaded successfully!")

# ================== 3. IMAGE PROCESSING ==================
def load_and_preprocess_image(image_path, target_size=(256, 256)):
    """Load and preprocess an image to match the generator input."""
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Normalize to [0,1]
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Dataset Paths for RICE1 Train Data
root_dir = r"C:\Users\YamiD\Downloads\GAN\RICE"
cloudy_folder = os.path.join(root_dir, "RICE1", "cloud")  # ✅ Train dataset path
output_folder = os.path.join(root_dir, "RICE1", "generated_train_2")  # ✅ New output path

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# ================== 4. GENERATE & SAVE IMAGES ==================
def generate_images(cloudy_folder, output_folder):
    """Generate and save cloud-free images using the trained GAN."""
    image_files = sorted(os.listdir(cloudy_folder))
    
    for img_name in image_files:
        try:
            img_path = os.path.join(cloudy_folder, img_name)
            img_tensor = load_and_preprocess_image(img_path)  # Load & preprocess image

            # Generate cloud-free image
            generated_image = generator.predict(img_tensor)[0]  # Remove batch dimension
            generated_image = (generated_image * 255).astype(np.uint8)  # Convert to uint8

            # Save generated image
            output_img_path = os.path.join(output_folder, img_name)
            array_to_img(generated_image).save(output_img_path)
            print(f"✅ Saved: {output_img_path}")
        
        except Exception as e:
            print(f"⚠️ Error processing {img_name}: {e}")

# Generate images for RICE1 Train dataset
generate_images(cloudy_folder, output_folder)

print("✅ Image generation for RICE1 Train completed!")

# ================== 5. VISUALIZE SAMPLE RESULTS ==================
def display_sample_results(sample_images=3):
    """Display side-by-side comparison of cloudy, generated, and overlay images."""
    fig, axes = plt.subplots(sample_images, 3, figsize=(10, 3 * sample_images))

    image_files = sorted(os.listdir(cloudy_folder))[:sample_images]  # Select a few images
    
    for i, img_name in enumerate(image_files):
        try:
            # Load and resize images
            cloudy_img = load_img(os.path.join(cloudy_folder, img_name), target_size=(256, 256))
            generated_img = load_img(os.path.join(output_folder, img_name))

            axes[i, 0].imshow(cloudy_img)
            axes[i, 0].set_title("Cloudy Image")
            axes[i, 1].imshow(generated_img)
            axes[i, 1].set_title("Generated Image")
            
            # Convert images to numpy arrays and overlay them
            cloudy_array = np.array(cloudy_img)
            generated_array = np.array(generated_img)

            overlay_img = (cloudy_array * 0.5 + generated_array * 0.5).astype(np.uint8)
            axes[i, 2].imshow(overlay_img)
            axes[i, 2].set_title("Overlay Comparison")

            for ax in axes[i]:
                ax.axis("off")

        except Exception as e:
            print(f"⚠️ Error displaying {img_name}: {e}")

    plt.tight_layout()
    plt.show()

# Display a few test images
display_sample_results()
