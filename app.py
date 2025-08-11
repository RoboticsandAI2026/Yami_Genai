import streamlit as st
import tensorflow as tf
import numpy as np
import os
import random
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
import cv2
from tensorflow.keras.preprocessing import image as keras_image
import smtplib, ssl
from email.message import EmailMessage
from win10toast import ToastNotifier
from keras.config import enable_unsafe_deserialization
import matplotlib.pyplot as plt
import re
from collections import Counter
from torchvision import transforms

# ==== CONFIG ====
st.set_page_config("Multiple-Model Forest Monitoring", layout="centered")
enable_unsafe_deserialization()

# ==== Paths ====
GAN_GENERATOR_PATH = r"C:\Gen AI project\GAN training\models_2\generator_rice1_latest.keras"
WILDLIFE_MODEL_PATH = r"C:\Gen AI project\wildlife classification\final_wildlife_model_1.keras"
FOREST_MODEL_PATH = r"C:\Gen AI project\forest change classification\foresst_conservation.keras"
FEDERATED_MODEL_PATH = r"C:\Gen AI project\wildlife classification\models\federated_model\wildlife_model_round_50 _1.pth"
WILDLIFE_TRAIN_DIR = r"C:\Gen AI project\wildlife classification\grouped_data\train"
FOREST_TRAIN_DIR = r"C:\Gen AI project\forest change classification\split_data\train"

# ==== Output Folders ====
OUTPUT_REALTIME = Path(r"C:\Gen AI project\multi_model_system\multi_model_outputs")
OUTPUT_WILDLIFE = Path(r"C:\Gen AI project\multi_model_system\wildlife_outputs")
OUTPUT_FOREST = Path(r"C:\Gen AI project\multi_model_system\forest_outputs")
OUTPUT_REALTIME.mkdir(parents=True, exist_ok=True)
OUTPUT_WILDLIFE.mkdir(parents=True, exist_ok=True)
OUTPUT_FOREST.mkdir(parents=True, exist_ok=True)

# ==== Email Config ====
GMAIL_SENDER = "yamichowdary25@gmail.com"
GMAIL_PASSWORD = "qejo jswi ndpr zipd"
RECEIVER_EMAIL = "yamichowdary2505@gmail.com"

# ==== Image Sizes ====
IMG_SIZE_GAN = (256, 256)
IMG_SIZE_WILDLIFE = (128, 128)
IMG_SIZE_FEDERATED = (224, 224)  # Standard size for many PyTorch models
IMG_SIZE_FOREST = (256, 256)

# ==== Label Standardization Function ====
def standardize_label(label):
    """Clean and standardize classification labels."""
    if not isinstance(label, str):
        return "Unknown"
        
    # Remove parenthetical notes (these are causing confusion)
    clean_label = re.sub(r'\s*\([^)]*\)', '', label).strip()
    
    # Convert to lowercase for comparison
    clean_label = clean_label.lower()
    
    # Define mapping to standardize similar categories
    label_mapping = {
        # Core wildlife categories
        'big_cats': 'big_cats',
        'birds': 'birds',
        'primates': 'primates',
        'small_mammals': 'small_mammals',
        'other_mammals': 'other_mammals',
        'sea_life': 'sea_life',
        'reptiles': 'reptiles',
        'rodents': 'rodents',
        'insects': 'insects',
        'amphibians': 'amphibians',
        'crustaceans': 'crustaceans',
        'poacher': 'human',
        'human': 'human'
    }
    
    # Find the closest matching category
    for key in label_mapping:
        if key in clean_label:
            return label_mapping[key]
            
    return clean_label

def detect_fog_smoke_cloud(image_tensor):
    """
    Detects low-contrast images that may contain fog, smoke, or clouds.
    """
    gray = tf.image.rgb_to_grayscale(image_tensor)
    stddev = tf.math.reduce_std(gray)
    mean = tf.math.reduce_mean(gray)

    # Tune these as needed
    if stddev < 0.05 or mean > 0.8:
        return True  # Likely fog/smoke/cloud
    return False

def debug_and_fix_class_labels(original_classes):
    """Print and fix class label issues"""
    
    # Print original classes
    print("Original wildlife classes:", original_classes)
    
    # Clean up class labels - remove parentheses and standardize
    cleaned_classes = [standardize_label(cls) for cls in original_classes]
    print("Cleaned wildlife classes:", cleaned_classes)
    
    # Check for duplicate classes after cleaning
    if len(set(cleaned_classes)) < len(cleaned_classes):
        print("WARNING: Duplicate class names after cleaning. This will cause problems.")
        # Create a frequency count
        duplicates = [item for item, count in Counter(cleaned_classes).items() if count > 1]
        print(f"Duplicate classes: {duplicates}")
        # Create a mapping from original to cleaned, ensuring uniqueness
        class_mapping = {}
        for i, (orig, cleaned) in enumerate(zip(original_classes, cleaned_classes)):
            if cleaned in class_mapping.values():
                # If duplicate, append index to make unique
                class_mapping[orig] = f"{cleaned}_{i}"
            else:
                class_mapping[orig] = cleaned
        print("Using disambiguated class mapping:", class_mapping)
        return original_classes, class_mapping
    else:
        print("Using cleaned class labels")
        # Create a direct mapping
        class_mapping = {orig: cleaned for orig, cleaned in zip(original_classes, cleaned_classes)}
        return cleaned_classes, class_mapping

# ==== Model Confidence Calibration ====
def calibrate_confidence_scores():
    """Create calibration factors for each model to normalize confidence scores"""
    # These values should be tuned based on model performance
    st.session_state.calibration_factors = {
        'cnn': 1.0,       # No adjustment to CNN confidence
        'federated': 0.9  # Slightly reduce federated confidence (tends to be overconfident)
    }
    
    # Define class-specific model preferences based on model strengths
    st.session_state.class_model_preference = {
        'crustaceans': 'cnn',    # Prefer CNN for crustaceans
        'reptiles': 'federated', # Prefer federated for reptiles 
        'insects': 'cnn',        # Prefer CNN for insects
        'birds': 'cnn',          # Prefer CNN for birds
        'primates': 'federated', # Prefer federated for primates
        'rodents': 'federated',  # Prefer federated for rodents
        'big_cats': 'cnn',       # Prefer CNN for big cats
        'sea_life': 'cnn',       # Prefer CNN for sea life
    }
    
    # Log the calibration
    print(f"Using confidence calibration: CNN factor = {st.session_state.calibration_factors['cnn']}, Federated factor = {st.session_state.calibration_factors['federated']}")
    print(f"Class-specific model preferences: {st.session_state.class_model_preference}")

# ==== Load TensorFlow Models ====
@st.cache_resource
def load_tensorflow_models():
    wildlife_model = tf.keras.models.load_model(WILDLIFE_MODEL_PATH)
    forest_model = tf.keras.models.load_model(FOREST_MODEL_PATH)
    gan_generator = tf.keras.models.load_model(GAN_GENERATOR_PATH, compile=False)
    return wildlife_model, forest_model, gan_generator

wildlife_model, forest_model, gan_generator = load_tensorflow_models()

# Fixed class list to match training class-to-index mapping
original_wildlife_classes = [
    'bears', 'big_cats', 'birds', 'crustaceans', 'hoofed',
    'insects', 'other_mammals', 'poacher', 'primates',
    'reptiles', 'rodents', 'sea_life'
]


wildlife_classes = original_wildlife_classes.copy()
class_mapping = {cls: cls for cls in original_wildlife_classes}

print("✅ Using fixed class list:")
for i, cls in enumerate(original_wildlife_classes):
    print(f"  Index {i}: {cls}")


# Initialize model calibration
calibrate_confidence_scores()

# Display class mapping at startup
print("Class mapping dictionary:")
for orig, new in class_mapping.items():
    print(f"  {orig} -> {new}")

# ==== Load PyTorch Federated Learning Model ====
@st.cache_resource
def load_federated_model():
    try:
        # Match the exact architecture used in training
        class WildlifeCNN(nn.Module):
            def __init__(self, num_classes=12):
                super(WildlifeCNN, self).__init__()
                
                # First convolutional block - match training exactly
                self.conv1 = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout(0.25)
                )

                # Second convolutional block
                self.conv2 = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout(0.25)
                )

                # Third convolutional block
                self.conv3 = nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout(0.25)
                )

                # Fourth convolutional block
                self.conv4 = nn.Sequential(
                    nn.Conv2d(256, 512, kernel_size=3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout(0.25)
                )

                # Calculate input size dynamically
                fc_input_size = 512 * (IMG_SIZE_FEDERATED[0] // 16) * (IMG_SIZE_FEDERATED[1] // 16)
                
                # Fully connected layers
                self.fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(fc_input_size, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, num_classes)
                )

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                x = self.conv4(x)
                x = self.fc(x)
                return x
                
        # Create model instance
        model = WildlifeCNN()
        
        # Print model summary to debug architecture
        print(f"Loading federated model from: {FEDERATED_MODEL_PATH}")
        
        # Load the model with error handling
        try:
            state_dict = torch.load(FEDERATED_MODEL_PATH, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            print("Federated model loaded successfully!")
        except Exception as e:
            print(f"Error loading state dict: {e}")
            # Try loading with strict=False as fallback
            model.load_state_dict(state_dict, strict=False)
            print("Federated model loaded with strict=False")
            
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading federated model: {e}")
        print(f"Detailed error: {e}")
        return None

# Load the federated model
federated_model = load_federated_model()

# ==== Enhanced Image Preprocessing Functions ====
def enhance_animal_contrast(img):
    """Enhance contrast to make animals stand out from background"""
    img_array = np.array(img)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    
    # Split the LAB image into L, A, and B channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge the CLAHE enhanced L-channel with the A and B channels
    limg = cv2.merge((cl, a, b))
    
    # Convert from LAB back to RGB
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    # Return enhanced image as PIL Image
    return Image.fromarray(enhanced)

# ==== Water Detection Function ====
def detect_water(img):
    """Detect if the image contains a water body like a pond or lake"""
    img_array = np.array(img)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Improved HSV ranges for water detection (blue-ish colors)
    lower_blue = np.array([90, 40, 40])
    upper_blue = np.array([140, 255, 255])
    
    # Create mask for water
    water_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
    
    # Calculate percentage of water in image
    water_ratio = np.count_nonzero(water_mask) / (img_array.shape[0] * img_array.shape[1])
    
    return water_ratio > 0.15, water_ratio  # Return both boolean and ratio

# ==== Enhanced Vegetation Detection ====
def detect_vegetation(img):
    """
    Enhanced vegetation detection with better differentiation between 
    actual forests and scattered vegetation.
    """
    img_array = np.array(img.resize(IMG_SIZE_FOREST))
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # More precise green ranges for different types of vegetation
    # Dense forest vegetation (darker green)
    forest_mask = cv2.inRange(hsv, (35, 70, 30), (85, 255, 180))
    
    # General vegetation (broader green range)
    vegetation_mask = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
    
    # Calculate coverage percentages
    forest_ratio = np.count_nonzero(forest_mask) / (IMG_SIZE_FOREST[0] * IMG_SIZE_FOREST[1])
    vegetation_ratio = np.count_nonzero(vegetation_mask) / (IMG_SIZE_FOREST[0] * IMG_SIZE_FOREST[1])
    
    # Analyze vegetation distribution - forests have more consistent distribution
    # Apply morphological operations to detect clusters
    kernel = np.ones((5, 5), np.uint8)
    vegetation_closed = cv2.morphologyEx(vegetation_mask, cv2.MORPH_CLOSE, kernel)
    
    # Count connected components (vegetation clusters)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(vegetation_closed)
    
    # Calculate cluster metrics
    large_clusters = 0
    max_cluster_size = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 500:
            large_clusters += 1
            max_cluster_size = max(max_cluster_size, stats[i, cv2.CC_STAT_AREA])
    
    max_cluster_ratio = max_cluster_size / (IMG_SIZE_FOREST[0] * IMG_SIZE_FOREST[1]) if large_clusters > 0 else 0
    
    # Calculate texture complexity (forests have more texture)
    grey = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    veg_texture = cv2.bitwise_and(grey, grey, mask=vegetation_mask)
    if np.any(veg_texture):
        laplacian = cv2.Laplacian(veg_texture[veg_texture > 0], cv2.CV_64F)
        texture_variance = np.var(laplacian)
    else:
        texture_variance = 0
    
    # Decision logic combining all features
    is_forest = False
    forest_confidence = 0.0
    
    # Strong indicators of forest
    if forest_ratio > 0.4 and large_clusters > 0 and max_cluster_ratio > 0.3:
        is_forest = True
        forest_confidence = min(1.0, forest_ratio * 1.5)
    # Moderate forest indicators
    elif vegetation_ratio > 0.3 and large_clusters > 0:
        is_forest = True
        forest_confidence = min(1.0, vegetation_ratio * 1.2)
    # Weak indicators - scattered vegetation or small patches
    elif vegetation_ratio > 0.15:
        is_forest = True
        forest_confidence = vegetation_ratio * 0.8
    else:
        is_forest = False
        forest_confidence = vegetation_ratio * 0.5
    
    # Log detection details
    print(f"Forest detection metrics: forest_ratio={forest_ratio:.2f}, vegetation_ratio={vegetation_ratio:.2f}, "
          f"large_clusters={large_clusters}, max_cluster_ratio={max_cluster_ratio:.2f}, "
          f"texture_variance={texture_variance:.1f}")
    print(f"Forest detection result: {is_forest} (confidence: {forest_confidence:.2f})")
    
    return is_forest, forest_confidence

# ==== Enhanced Content Detection ====
def detect_wildlife_in_image(img):
    """
    More robust wildlife detection that better identifies farm animals and wildlife
    with reduced false negatives for domesticated animals.
    """
    img_array = np.array(img)
    
    # Method 1: Enhanced Edge-based detection with better animal shape recognition
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 140)  # Slightly more sensitive parameters
    
    # Find contours with improved parameters for animal shapes
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    wildlife_indicator = 0
    potential_wildlife_contours = []
    
    # Analyze contours for animal-like shapes (both wild and domestic)
    for contour in contours:
        area = cv2.contourArea(contour)
        # Modified size range to better detect larger animals like cows
        if 300 < area < 15000:  
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                # Calculate shape metrics with parameters tuned for quadrupeds
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Modified shape characteristics to include farm animals
                if 0.1 < circularity < 0.95 and 0.25 < aspect_ratio < 3.5:
                    wildlife_indicator += 1
                    potential_wildlife_contours.append(contour)
    
    # Method 2: Improved texture analysis for fur/hide patterns
    if len(potential_wildlife_contours) > 0:
        # Create a mask of potential animal areas
        wildlife_mask = np.zeros_like(gray)
        cv2.drawContours(wildlife_mask, potential_wildlife_contours, -1, 255, -1)
        
        # Check texture complexity in these areas
        texture_regions = cv2.bitwise_and(gray, gray, mask=wildlife_mask.astype(np.uint8))
        if np.any(texture_regions):
            laplacian = cv2.Laplacian(texture_regions[texture_regions > 0], cv2.CV_64F)
            texture_variance = np.var(laplacian)
            # Adjusted texture threshold to better detect animal hide/fur
            if texture_variance > 80:  # Lower threshold to detect smoother textures of some animals
                wildlife_indicator += 2
    
    # Method 3: CNN probability check with modified threshold for domesticated animals
    img_resized = np.expand_dims(np.array(img.resize(IMG_SIZE_WILDLIFE)) / 255.0, axis=0)
    preds = wildlife_model.predict(img_resized)[0]
    top_class_idx = np.argmax(preds)
    top_conf = np.max(preds)
    
    # Don't exclude 'poacher'/'human' class from wildlife detection
    # Instead check if it's the dominant pattern and significantly above threshold
    human_detected = False
    if top_class_idx == original_wildlife_classes.index('poacher') and top_conf > 0.6:
        human_detected = True
        # Reduce wildlife indicator but don't reset completely
        wildlife_indicator = max(0, wildlife_indicator - 2) 
    
    # Modified CNN confidence threshold - more lenient to reduce false negatives
    if top_conf > 0.4:  # Lowered threshold from 0.5
        wildlife_indicator += 2
        if top_conf > 0.6:  # Add more weight to high confidence predictions
            wildlife_indicator += 1
    
    # Method 4: Enhanced animal pattern detection
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Calculate color variance in potential wildlife areas
    if len(potential_wildlife_contours) > 0:
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, potential_wildlife_contours, -1, 255, -1)
        
        if np.any(mask):
            # Check if there are distinctive patterns in each channel
            h_var = np.var(hsv[:,:,0][mask > 0])
            s_var = np.var(hsv[:,:,1][mask > 0])
            v_var = np.var(hsv[:,:,2][mask > 0])
            
            # Adjusted thresholds for both domestic and wild animal patterns
            if (h_var > 80) or (s_var > 150) or (v_var > 350):  # Lower thresholds
                wildlife_indicator += 1
    
    # Method 5: NEW - Check for animal-specific colors (like black and white pattern of Holstein cows)
    if len(potential_wildlife_contours) > 0:
        # Create mask and extract only the regions with potential animals
        animal_mask = np.zeros_like(gray)
        cv2.drawContours(animal_mask, potential_wildlife_contours, -1, 255, -1)
        
        # Convert to binary for pattern analysis
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        animal_binary = cv2.bitwise_and(binary, binary, mask=animal_mask)
        
        if np.any(animal_binary):
            # Count transitions from white to black (good indicator of patterns like cow spots)
            transitions = 0
            prev_value = 0
            for i in range(1, animal_binary.shape[0], 10):  # Sample rows
                for j in range(1, animal_binary.shape[1]-1):
                    if animal_binary[i,j] != animal_binary[i,j+1] and animal_mask[i,j] > 0:
                        transitions += 1
            
            # Normalize by the area
            area = np.count_nonzero(animal_mask)
            transition_density = transitions / area if area > 0 else 0
            
            # High transition density indicates patterned animals (like cows)
            if transition_density > 0.01:
                wildlife_indicator += 2
                print(f"Detected animal patterns with transition density: {transition_density:.4f}")
    
    # Final decision with adjusted threshold to reduce false negatives
    has_wildlife = wildlife_indicator >= 2.5  # Lowered from 4
    confidence = min(1.0, wildlife_indicator / 8)
    
    # Less aggressive false positive filtering for vegetation
    if has_wildlife:
        # Check if the "wildlife" is actually just tree branches
        green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        potential_wildlife_pixels = np.count_nonzero(wildlife_mask) if len(potential_wildlife_contours) > 0 else 0
        if potential_wildlife_pixels > 0:
            green_wildlife_ratio = np.count_nonzero(cv2.bitwise_and(green_mask, wildlife_mask)) / potential_wildlife_pixels
            
            # Higher threshold to avoid rejecting animals in grassland
            if green_wildlife_ratio > 0.85:  # More stringent, was 0.7
                has_wildlife = False
                confidence *= 0.3
                print("Rejected wildlife detection: likely vegetation")
    
    # Print detailed detection info
    detection_sources = []
    if len(potential_wildlife_contours) > 2:
        detection_sources.append("shape analysis")
    if top_conf > 0.4:
        detection_sources.append(f"CNN ({top_conf:.2f})")
    if transitions > 50 if 'transitions' in locals() else False:
        detection_sources.append("pattern detection")
    
    sources_text = " and ".join(detection_sources) if detection_sources else "insufficient evidence"
    print(f"Wildlife detection: {has_wildlife} (confidence: {confidence:.2f}, based on {sources_text})")
    
    return has_wildlife, confidence, potential_wildlife_contours

def detect_image_content(img, wildlife_model):
    """
    Comprehensive image content detection that combines multiple methods
    for more accurate classification of image content.
    """
    # Check for forest/vegetation with improved detection
    has_forest, forest_ratio = detect_vegetation(img)

    # Check for wildlife with enhanced detection
    has_wildlife, wildlife_confidence, wildlife_contours = detect_wildlife_in_image(img)

    # Water detection
    water_present, water_ratio = detect_water(img)
    
    # Dominant color analysis
    dominant_colors = get_dominant_colors(img)
    
    # Check if image is too dark or low quality
    img_array = np.array(img)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    is_low_quality = brightness < 40 or contrast < 20
    
    # Enhanced scene understanding
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Detect sky
    sky_mask = cv2.inRange(hsv, (80, 0, 120), (140, 80, 255))
    sky_ratio = np.count_nonzero(sky_mask) / (img_array.shape[0] * img_array.shape[1])
    has_sky = sky_ratio > 0.15
    
    # Green vegetation ratio
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    green_ratio = np.count_nonzero(green_mask) / (img_array.shape[0] * img_array.shape[1])
    
    # Environment classification
    environment_type = "Unknown"
    if green_ratio > 0.5:
        environment_type = "Dense Forest"
    elif green_ratio > 0.3:
        environment_type = "Woodland"
    elif green_ratio > 0.1:
        environment_type = "Sparse Vegetation"
    elif water_ratio > 0.3:
        environment_type = "Water Body"
    elif sky_ratio > 0.5:
        environment_type = "Open Sky"
    else:
        environment_type = "Other Terrain"
    
    return {
        "has_forest": has_forest,
        "forest_confidence": forest_ratio,
        "has_wildlife": has_wildlife,
        "wildlife_confidence": wildlife_confidence,
        "water_present": water_present,
        "water_ratio": water_ratio,
        "green_ratio": green_ratio,
        "sky_ratio": sky_ratio,
        "environment_type": environment_type,
        "dominant_colors": dominant_colors,
        "image_quality": {
            "brightness": brightness,
            "contrast": contrast,
            "is_low_quality": is_low_quality
        }
    }

# ==== Enhanced Vegetation Health Analysis ====
def analyze_vegetation_health(img):
    """Analyze vegetation health and development potential"""
    img_array = np.array(img)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Enhanced vegetation indices
    # Green vegetation
    green_mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
    
    # Stressed vegetation (yellowish)
    stressed_mask = cv2.inRange(hsv, (20, 50, 50), (35, 255, 255))
    
    # Dead vegetation (brownish)
    dead_mask = cv2.inRange(hsv, (0, 30, 30), (20, 200, 200))
    
    # Calculate ratios
    total_pixels = img_array.shape[0] * img_array.shape[1]
    healthy_ratio = np.count_nonzero(green_mask) / total_pixels
    stressed_ratio = np.count_nonzero(stressed_mask) / total_pixels
    dead_ratio = np.count_nonzero(dead_mask) / total_pixels
    
    # Calculate overall health score (0-100)
    health_score = int((healthy_ratio * 100) - (stressed_ratio * 30) - (dead_ratio * 80))
    health_score = max(0, min(100, health_score))
    
    # Development potential based on health and disturbance signs
    # Check for straight lines/edges (man-made structures)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    # Presence of straight lines indicates development
    development_indicators = 0
    if lines is not None:
        development_indicators = len(lines)
    
    # Development potential score
    development_potential = min(100, (development_indicators * 2) + (dead_ratio * 50) + (stressed_ratio * 30) + ((1 - healthy_ratio) * 20))
    
    # Health categories
    if health_score >= 80:
        health_category = "Excellent"
    elif health_score >= 60:
        health_category = "Good" 
    elif health_score >= 40:
        health_category = "Moderate"
    elif health_score >= 20:
        health_category = "Poor"
    else:
        health_category = "Critical"
    
    # Development potential categories
    if development_potential >= 75:
        dev_category = "High"
    elif development_potential >= 50:
        dev_category = "Moderate"
    elif development_potential >= 25:
        dev_category = "Low"
    else:
        dev_category = "Very Low"
    
    return {
        "health_score": health_score,
        "health_category": health_category,
        "development_potential": development_potential,
        "development_category": dev_category,
        "healthy_ratio": healthy_ratio,
        "stressed_ratio": stressed_ratio,
        "dead_ratio": dead_ratio
    }

# ==== Helper Function to Get Dominant Colors ====
def get_dominant_colors(img, k=3):
    """Extract the k dominant colors from an image"""
    # Resize image for faster processing
    small_img = img.resize((50, 50))
    pixels = np.float32(np.array(small_img).reshape(-1, 3))
    
    # Use k-means to find dominant colors
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Count pixel frequencies
    counts = np.bincount(labels.flatten())
    
    # Sort by frequency
    sorted_indices = np.argsort(counts)[::-1]
    
    # Return RGB dominant colors sorted by frequency
    return centers[sorted_indices].astype(np.uint8)

# ==== Cloud Detection ====
def is_clouded(img):
    """
    Improved cloud detection that properly distinguishes between actual atmospheric clouds
    and visual artifacts like slight blur, dust, or low contrast areas.
    Returns True if the image contains actual clouds, False otherwise.
    """
    # Convert to array
    img_array = np.array(img)
    
    # Convert to HSV for better cloud detection
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Extract HSV channels
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    
    # Get grayscale for texture analysis
    gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # IMPROVED: Texture analysis to distinguish clouds from artifacts
    # Calculate local standard deviation as a measure of texture
    kernel_size = 15
    mean_img = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0)
    squared_diff = cv2.GaussianBlur(np.square(gray_img - mean_img), (kernel_size, kernel_size), 0)
    local_std = np.sqrt(np.maximum(squared_diff, 0))
    
    # Calculate gradient magnitude for edge detection (clouds have soft edges)
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # IMPROVED: Multiple conditions for cloud detection with better thresholds
    # 1. White clouds - high value, very low saturation, medium-low texture
    white_cloud_mask = (s < 30) & (v > 200) & (local_std < 15) & (gradient_magnitude < 25)
    
    # 2. Grey clouds - medium-high value, low saturation, medium-low texture
    grey_cloud_mask = (s < 25) & (v > 130) & (v < 210) & (local_std < 12) & (gradient_magnitude < 20)
    
    # IMPROVED: Filter out dust and blur artifacts
    # Dust typically has higher local texture variation despite low saturation
    dust_mask = (s < 40) & (local_std > 12) & (gradient_magnitude > 15)
    
    # Blur has very low gradient everywhere, not just in cloud regions
    blur_ratio = np.count_nonzero(gradient_magnitude < 10) / (img_array.shape[0] * img_array.shape[1])
    is_blurry = blur_ratio > 0.7
    
    # IMPROVED: Check image regions - clouds usually appear in upper portions of image
    # Create a weight mask that gives higher importance to upper part of image
    height, width = gray_img.shape
    y_coords = np.linspace(0, 1, height)[:, np.newaxis] * np.ones((height, width))
    upper_weight = 1.0 - y_coords  # Weight decreases from top to bottom
    
    # Apply positional weighting to cloud masks
    weighted_white_cloud = white_cloud_mask * upper_weight
    weighted_grey_cloud = grey_cloud_mask * upper_weight
    
    # Calculate percentage of cloudy pixels for each type with regional weighting
    white_cloud_ratio = np.sum(weighted_white_cloud) / np.sum(upper_weight)
    grey_cloud_ratio = np.sum(weighted_grey_cloud) / np.sum(upper_weight)
    dust_ratio = np.count_nonzero(dust_mask) / (img_array.shape[0] * img_array.shape[1])
    
    # IMPROVED: Combined cloud detection logic with artifact rejection
    total_cloud_ratio = white_cloud_ratio + grey_cloud_ratio
    
    # Calculate overall image quality metrics
    brightness = np.mean(v)
    contrast = np.std(gray_img)
    
    # IMPROVED: Sky detection - clouds typically appear against blue sky
    sky_hue_mask = ((h > 90) & (h < 140)) & (s > 20) & (s < 150) & (v > 150)
    sky_ratio = np.count_nonzero(sky_hue_mask) / (img_array.shape[0] * img_array.shape[1])
    
    # Print detailed cloud detection metrics
    print(f"Cloud detection metrics: white={white_cloud_ratio:.3f}, grey={grey_cloud_ratio:.3f}, " 
          f"dust={dust_ratio:.3f}, total={total_cloud_ratio:.3f}, sky={sky_ratio:.3f}")
    print(f"Image quality: brightness={brightness:.1f}, contrast={contrast:.1f}, blur_ratio={blur_ratio:.3f}")
    
    # IMPROVED: Decision logic with better artifact rejection
    is_cloud = False
    message = "No clouds detected"
    
    # First check if the image is just blurry overall
    if is_blurry and contrast < 40:
        is_cloud = False
        message = "Image appears blurry, but no actual clouds detected"
    # Check for significant cloud presence with positional weighting
    elif total_cloud_ratio > 0.15 and sky_ratio > 0.05:
        is_cloud = True
        message = "Significant cloud coverage detected"
    # Check for light cloud coverage
    elif total_cloud_ratio > 0.08 and sky_ratio > 0.1 and not is_blurry:
        is_cloud = True
        message = "Light cloud coverage detected"
    # Very bright overall with low contrast might be overexposed, not cloudy
    elif brightness > 190 and contrast < 30:
        is_cloud = False
        message = "Image appears overexposed, but not necessarily cloudy"
    # High dust but low cloud ratio indicates artifacts, not clouds
    elif dust_ratio > 0.1 and total_cloud_ratio < 0.1:
        is_cloud = False
        message = "Visual artifacts or dust detected, but not atmospheric clouds"
    else:
        is_cloud = False
        message = "No significant clouds detected"
    
    return is_cloud, total_cloud_ratio, message

def remove_clouds(img, gan_generator):
    """
    Improved cloud removal using GAN with better preprocessing and 
    post-processing for more natural results.
    """
    # Preserve original image for blending
    original = np.array(img)
    
    # Prepare image for GAN (normalize to [0,1])
    img_resized = img.resize(IMG_SIZE_GAN)
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
    
    # Generate cloud-free image
    result = gan_generator.predict(img_array)[0]
    
    # Convert back to [0,255] range
    result_image = (result * 255).astype(np.uint8)
    
    # Resize back to original dimensions if needed
    if img.size != IMG_SIZE_GAN:
        result_image = cv2.resize(result_image, (img.size[0], img.size[1]))
    
    # Create a cloud mask from original image
    _, _, mask_message = is_clouded(img)
    hsv = cv2.cvtColor(original, cv2.COLOR_RGB2HSV)
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    cloud_mask = ((s < 40) & (v > 180)).astype(np.float32)
    
    # Smooth the mask for better blending
    cloud_mask = cv2.GaussianBlur(cloud_mask, (25, 25), 0)
    cloud_mask = np.stack([cloud_mask, cloud_mask, cloud_mask], axis=2)
    
    # Blend original and GAN-processed image based on cloud mask
    # Areas with clouds (mask=1) will use GAN result, clear areas (mask=0) use original
    blended = original * (1 - cloud_mask) + result_image * cloud_mask
    
    # Enhance contrast of the result
    lab = cv2.cvtColor(blended.astype(np.uint8), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(enhanced.astype(np.uint8)), mask_message

# ==== SIMPLIFIED Wildlife Classification with Image Enhancement ====
def classify_wildlife(img):
    """Classify wildlife using CNN model with standardized labels and farm animal detection"""
    # Create input for model
    img_array = np.expand_dims(np.array(img.resize(IMG_SIZE_WILDLIFE)) / 255.0, axis=0)
    
    # Get predictions
    preds = wildlife_model.predict(img_array)[0]
    
    # Get top 2 predictions
    top_indices = np.argsort(preds)[-2:][::-1]
    
    # Get original class names
    top_class_original = original_wildlife_classes[top_indices[0]]
    second_class_original = original_wildlife_classes[top_indices[1]]
    
    # Map to standardized names
    top_class = class_mapping[top_class_original]
    second_class = class_mapping[second_class_original]
    
    # Get confidence scores
    top_conf = preds[top_indices[0]]
    second_conf = preds[top_indices[1]]
    
    # Farm animal detection logic - if hoofed or other_mammals with specific patterns
    hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
    
    # Check specifically for cow-like patterns (black and white)
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Look for black-white transitions (cow spots)
    transitions = 0
    for i in range(1, binary.shape[0]-1, 10):
        for j in range(1, binary.shape[1]-1, 10):
            if binary[i,j] != binary[i,j+1]:
                transitions += 1
    
    # Calculate pattern density
    transition_density = transitions / (binary.shape[0] * binary.shape[1])
    
    # If it looks like a farm animal (cow pattern)
    if (top_class in ['hoofed', 'other_mammals'] and transition_density > 0.01) or \
       (transition_density > 0.015):  # Strong pattern independent of class
        # Override with farm animal category
        if top_conf < 0.7:  # Only override if not very confident
            print(f"Detected farm animal pattern (transitions: {transition_density:.4f}), adjusting classification")
            top_class = "farm_animal"  # New category
            top_conf = max(top_conf, 0.65)  # Ensure reasonable confidence
    
    # Calculate uncertainty
    uncertainty = top_conf - second_conf
    
    return top_class, top_conf, second_class, second_conf, uncertainty, top_class_original


# ==== SIMPLIFIED Federated Model Classification ====
def classify_federated(img):
    """Classify wildlife using federated model with standardized labels"""
    if federated_model is None:
        return None, 0.0, None, 0.0, 1.0, None
    
    # Use the EXACT same preprocessing as during training
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE_FEDERATED),
        transforms.ToTensor(),
        # Use the exact same normalization as during training
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Prepare image
    img_tensor = transform(img).unsqueeze(0)
    
    # Get predictions
    with torch.no_grad():
        try:
            outputs = federated_model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get top 2 predictions
            confidences, indices = torch.topk(probs, 2, dim=1)
            
            # Print prediction details for debugging
            top_class_idx = indices[0][0].item()
            print(f"Federated model prediction: Class {top_class_idx} with confidence {confidences[0][0].item():.4f}")
            
            # Get original class names
            top_class_original = original_wildlife_classes[top_class_idx]
            second_class_original = original_wildlife_classes[indices[0][1].item()]
            
            # Map to standardized names
            top_class = class_mapping[top_class_original]
            second_class = class_mapping[second_class_original]
            
            # Get confidence scores
            top_conf = confidences[0][0].item()
            second_conf = confidences[0][1].item()
            
            # Apply confidence calibration
            if 'calibration_factors' in st.session_state:
                calibration = st.session_state.calibration_factors.get('federated', 1.0)
                top_conf = min(1.0, top_conf * calibration)
            
            # Only trust the model if confidence is reasonable
            if top_conf < 0.5:
                print("⚠️ Federated model returned low confidence. Marking as uncertain.")
                return "Uncertain", top_conf, None, 0.0, 1.0, None
            
            # Calculate uncertainty (difference between top two predictions)
            uncertainty = top_conf - second_conf
            
            return top_class, top_conf, second_class, second_conf, uncertainty, top_class_original
        except Exception as e:
            print(f"Error during federated prediction: {e}")
            return "Error", 0.0, None, 0.0, 1.0, None

def update_session_calibration():
    """Update session state to add farm animal detection capabilities"""
    if 'class_model_preference' in st.session_state:
        # Add farm animal preference to existing class preferences
        st.session_state.class_model_preference['farm_animal'] = 'cnn'
        
        # Print updated preferences
        print("Updated class preferences to include farm animals:")
        print(st.session_state.class_model_preference)
    else:
        print("Warning: session state not initialized properly for class preferences")

# Call this at startup to ensure farm animal detection is available
update_session_calibration()

# ==== Enhanced Forest Classification ====
def analyze_forest(img):
    img_array = keras_image.img_to_array(img.resize(IMG_SIZE_FOREST)) / 255.0
    hsv = cv2.cvtColor(np.array(img.resize(IMG_SIZE_FOREST)), cv2.COLOR_RGB2HSV)
    
    # Improved color ranges
    green = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
    brown = cv2.inRange(hsv, (10, 50, 50), (30, 255, 200))
    dead_veg = cv2.inRange(hsv, (0, 0, 100), (20, 50, 200))  # Detect dead vegetation
    
    # Calculate color ratios
    green_ratio = np.count_nonzero(green) / (IMG_SIZE_FOREST[0] * IMG_SIZE_FOREST[1])
    brown_ratio = np.count_nonzero(brown) / (IMG_SIZE_FOREST[0] * IMG_SIZE_FOREST[1])
    dead_ratio = np.count_nonzero(dead_veg) / (IMG_SIZE_FOREST[0] * IMG_SIZE_FOREST[1])

    # Model prediction - use when available
    try:
        model_pred = np.argmax(forest_model.predict(np.expand_dims(img_array, axis=0))[0])
        model_classes = ["Conservation", "Deforestation", "Mixed", "Snag Trees"]
        model_result = model_classes[model_pred]
    except:
        model_result = None
    
    # Enhanced rule-based classification
    if green_ratio > 0.6:
        rule_result = "Conservation"
    elif brown_ratio > 0.5 or dead_ratio > 0.4:
        rule_result = "Snag Trees"
    elif green_ratio > 0.3 and brown_ratio > 0.2:
        rule_result = "Mixed Vegetation"
    elif green_ratio < 0.2 and brown_ratio < 0.2:
        rule_result = "Non-Forest Area"
    else:
        rule_result = "Potential Deforestation"
    
    # Use model prediction if available, otherwise rule-based
    return model_result if model_result else rule_result

# ==== Save Prediction ====
def save_output(folder, img, tag, wildlife_class, forest_class, conf, count):
    img.save(folder / f"img{count}.jpg")
    with open(folder / f"text{count}.txt", "w", encoding="utf-8") as f:
        f.write(f"Source: {tag}\n")
        f.write(f"Wildlife Class: {wildlife_class}\n")
        f.write(f"Forest Class: {forest_class}\n")
        f.write(f"Confidence: {conf:.2f}\n")

# ==== IMPROVED Combined Pipeline (Real-Time Upload) with Intelligent Content Detection ====
def process_combined(img, count):
    """Process an image with intelligent content-based classification"""
    
    # Show original image for reference
    st.markdown("### Input Image")
    st.image(img, caption="Original Input", use_column_width=True)
    
    # Cloud removal if needed
    is_cloudy, cloud_ratio, cloud_message = is_clouded(img)
    if is_cloudy:
        st.warning(f"☁️ {cloud_message} (ratio: {cloud_ratio:.2f})")
        # Store original image for comparison
        cloudy_img = img.copy()
        # Process with GAN
        img, message = remove_clouds(img, gan_generator)
        st.success(f"✅ Cloud removed: {message}")
    
        # Display before/after cloud removal
        col1, col2 = st.columns(2)
        with col1:
            st.image(cloudy_img, caption="Original Cloudy Image")
        with col2:
            st.image(img, caption="After Cloud Removal (GAN)")
    
    # Apply minimal preprocessing - just contrast enhancement
    img_enhanced = enhance_animal_contrast(img)
    
    # Show preprocessing in expander
    with st.expander("View Preprocessing"):
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original")
        with col2:
            st.image(img_enhanced, caption="Enhanced Contrast")
    
    # Smart content detection
    content_info = detect_image_content(img,wildlife_model)
    
    # Display content detection results
    st.markdown("### Content Detection")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Forest Detected:** {'Yes' if content_info['has_forest'] else 'No'} ({content_info['forest_confidence']:.2f})")
    with col2:
        st.markdown(f"**Wildlife Detected:** {'Yes' if content_info['has_wildlife'] else 'No'} ({content_info['wildlife_confidence']:.2f})")
    
    # Process based on detected content
    wildlife_output = None
    wildlife_confidence = 0.0
    forest_output = None
    forest_confidence = 0.0
    
    # Only run wildlife classification if wildlife detected
    if content_info['has_wildlife']:
        # Get raw predictions from each model
        cnn_class, cnn_confidence, cnn_second, cnn_second_conf, cnn_uncertainty, cnn_orig = classify_wildlife(img_enhanced)
        
        if federated_model is not None:
            fed_class, fed_confidence, fed_second, fed_second_conf, fed_uncertainty, fed_orig = classify_federated(img_enhanced)
        else:
            fed_class, fed_confidence, fed_second, fed_second_conf, fed_uncertainty, fed_orig = None, 0.0, None, 0.0, 1.0, None

        if fed_class != cnn_class and fed_confidence < 0.6:
            print("⚠️ Federated disagrees with CNN and confidence is weak. Ignoring.")
            fed_class = "Uncertain"
            fed_confidence = 0.0

        # Show raw prediction table
        st.markdown("### Wildlife Classification")
        raw_data = {
            "Model": ["CNN", "Federated"],
            "Original Prediction": [
                cnn_orig if cnn_orig else "N/A", 
                fed_orig if fed_orig else "N/A"
            ],
            "Standardized Class": [
                cnn_class,
                fed_class if fed_class else "N/A"
            ],
            "Confidence": [
                f"{cnn_confidence:.2f}",
                f"{fed_confidence:.2f}" if fed_class else "N/A"
            ]
        }
        st.table(raw_data)
        
        # Apply calibration to confidence scores
        calibrated_cnn_conf = cnn_confidence * st.session_state.calibration_factors['cnn']
        calibrated_fed_conf = fed_confidence * st.session_state.calibration_factors['federated'] if fed_confidence else 0.0
        
        # Apply class-specific biases if available
        class_bias_applied = False
        bias_explanation = ""
        
        if cnn_class in st.session_state.class_model_preference:
            preferred_model = st.session_state.class_model_preference[cnn_class]
            if preferred_model == 'cnn':
                calibrated_cnn_conf *= 1.1
                class_bias_applied = True
                bias_explanation = f"Applied +10% bias for CNN (preferred for {cnn_class})"
        
        if fed_class in st.session_state.class_model_preference:
            preferred_model = st.session_state.class_model_preference[fed_class]
            if preferred_model == 'federated':
                calibrated_fed_conf *= 1.1
                class_bias_applied = True
                bias_explanation += f"{'; ' if bias_explanation else ''}Applied +10% bias for Federated (preferred for {fed_class})"
        
        # Make the final model selection
        if cnn_confidence >= 0.6:
            wildlife_output = cnn_class
            wildlife_confidence = cnn_confidence
            classification_method = "CNN model (confident prediction ≥ 0.6)"
        elif fed_class:
            wildlife_output = fed_class
            wildlife_confidence = fed_confidence
            classification_method = "Federated model (used as fallback)"
        else:
            wildlife_output = cnn_class
            wildlife_confidence = cnn_confidence
            classification_method = "CNN model (fallback - low confidence)"
        
        # Show wildlife classification
        st.markdown(f"**Wildlife Classification:** `{wildlife_output}`")
        st.markdown(f"**Confidence:** `{wildlife_confidence:.2f}`")
        st.info(classification_method)
    
    # Only run forest classification if forest detected
    if content_info['has_forest']:
        st.markdown("### Forest Classification")
        
        # First, categorize the forest type
        forest_output = analyze_forest(img)
        
        # Then analyze vegetation health
        veg_health = analyze_vegetation_health(img)
        
        # Display forest type
        st.markdown(f"**Forest Type:** `{forest_output}`")
        
        # Display vegetation health metrics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Vegetation Health:** `{veg_health['health_category']}`")
            st.markdown(f"**Health Score:** `{veg_health['health_score']}/100`")
        with col2:
            st.markdown(f"**Development Potential:** `{veg_health['development_category']}`")
            st.markdown(f"**Development Score:** `{veg_health['development_potential']:.1f}/100`")
        
        # Create progress bars for health metrics
        st.markdown("#### Vegetation Health Breakdown")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Healthy Vegetation", f"{veg_health['healthy_ratio']*100:.1f}%")
            st.progress(veg_health['healthy_ratio'])
        with col2:
            st.metric("Stressed Vegetation", f"{veg_health['stressed_ratio']*100:.1f}%")
            st.progress(veg_health['stressed_ratio'])
        with col3:
            st.metric("Dead Vegetation", f"{veg_health['dead_ratio']*100:.1f}%")
            st.progress(veg_health['dead_ratio'])
        
        # Create health score gauge
        health_color = "green" if veg_health['health_score'] >= 70 else "orange" if veg_health['health_score'] >= 40 else "red"
        st.markdown(f"<div style='text-align:center'><h4>Vegetation Health Score</h4><div style='margin:0 auto; width:70%; height:30px; background:#eee; border-radius:15px;'><div style='width:{veg_health['health_score']}%; height:100%; background-color:{health_color}; border-radius:15px;'></div></div><p style='margin-top:5px'>{veg_health['health_score']}/100</p></div>", unsafe_allow_html=True)
        
        forest_confidence = content_info['forest_confidence']
    
    # No relevant content detected
    if not content_info['has_wildlife'] and not content_info['has_forest']:
        st.warning("⚠️ No forest or wildlife reliably detected in this image.")
        st.info("The image might contain other objects or scenes not covered by our models.")
    
    # Process successful
    st.success("✅ Processed.")
    
    # Final classification display
    st.markdown("### Final Classification Summary")
    
    summary_data = {
        "Content Type": ["Wildlife", "Forest"],
        "Detected": [
            "Yes" if content_info['has_wildlife'] else "No",
            "Yes" if content_info['has_forest'] else "No"
        ],
        "Classification": [
            wildlife_output if wildlife_output else "N/A",
            forest_output if forest_output else "N/A"
        ],
        "Confidence": [
            f"{wildlife_confidence:.2f}" if wildlife_output else "N/A",
            f"{forest_confidence:.2f}" if forest_output else "N/A"
        ]
    }
    st.table(summary_data)
    
    # Feedback collection
    with st.expander("Provide Feedback"):
        st.write("Was this classification correct?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Correct"):
                st.session_state.setdefault("correct_count", 0)
                st.session_state.correct_count += 1
                st.success(f"Thanks! Correct classifications: {st.session_state.correct_count}")
        with col2:
            if st.button("👎 Incorrect"):
                st.session_state.setdefault("incorrect_count", 0)
                st.session_state.incorrect_count += 1
                correct_class = st.text_input("What should it be? (Optional)")
                if correct_class:
                    st.session_state.setdefault("corrections", {})
                    if wildlife_output not in st.session_state.corrections:
                        st.session_state.corrections[wildlife_output] = {}
                    if correct_class not in st.session_state.corrections[wildlife_output]:
                        st.session_state.corrections[wildlife_output][correct_class] = 0
                    st.session_state.corrections[wildlife_output][correct_class] += 1
                st.error(f"Thanks for your feedback! Incorrect classifications: {st.session_state.incorrect_count}")
    
    # Save the results with proper content indication
    if wildlife_output and forest_output:
        save_output(OUTPUT_REALTIME, img, "Both", wildlife_output, forest_output, max(wildlife_confidence, forest_confidence), count)
    elif wildlife_output:
        save_output(OUTPUT_REALTIME, img, "Wildlife", wildlife_output, "-", wildlife_confidence, count)
    elif forest_output:
        save_output(OUTPUT_REALTIME, img, "Forest", "-", forest_output, forest_confidence, count)
    else:
        save_output(OUTPUT_REALTIME, img, "Unknown", "-", "-", 0.0, count)

# ==== Dataset Deployers ====
def deploy_dataset(folder, model_type):
    all_images = [os.path.join(root, f)
                  for root, _, files in os.walk(folder)
                  for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not all_images:
        st.warning("⚠️ No images found in dataset folder.")
        return

    sample = random.sample(all_images, min(5, len(all_images)))
    
    # Track misclassification indicators
    misclassification_risks = []

    for i, path in enumerate(sample):
        img = Image.open(path).convert("RGB")
        
        # Apply simple preprocessing
        img_enhanced = enhance_animal_contrast(img)
        
        # Detect content automatically instead of relying on model_type
        content_info = detect_image_content(img,wildlife_model)

        if model_type == "wildlife" or (model_type == "auto" and content_info['has_wildlife']):
            # Direct classification with simplified logic
            cnn_class, cnn_conf, _, _, _, cnn_orig = classify_wildlife(img_enhanced)
            
            if federated_model is not None:
                fed_class, fed_conf, _, _, _, fed_orig = classify_federated(img_enhanced)
            else:
                fed_class, fed_conf, fed_orig = None, 0.0, None
            
            # Apply calibration
            calibrated_cnn_conf = cnn_conf * st.session_state.calibration_factors['cnn']
            calibrated_fed_conf = fed_conf * st.session_state.calibration_factors['federated'] if fed_conf else 0.0
            
            # Apply class-specific biases if available
            if cnn_class in st.session_state.class_model_preference and st.session_state.class_model_preference[cnn_class] == 'cnn':
                calibrated_cnn_conf *= 1.1
            
            if fed_class in st.session_state.class_model_preference and st.session_state.class_model_preference[fed_class] == 'federated':
                calibrated_fed_conf *= 1.1
            
            # Simple selection - use the model with higher calibrated confidence
            if fed_class and calibrated_fed_conf > calibrated_cnn_conf + 0.05:
                pred = fed_class
                conf = fed_conf
                orig = fed_orig
                model_used = "Federated"
            else:
                pred = cnn_class
                conf = cnn_conf
                orig = cnn_orig
                model_used = "CNN"

            if fed_class == "Uncertain" or fed_conf < 0.5:
                risk = "High (Federated Unreliable)"
            else:
                risk = "Low" if fed_conf > 0.75 else "Medium"
    
            
            # Flag high risk classifications
            risk = "High" if conf < 0.6 else "Medium" if conf < 0.75 else "Low"
            
            # Track misclassification risks
            misclassification_risks.append((path, pred, orig, model_used, conf, risk))
                
            save_output(OUTPUT_WILDLIFE, img, "Dataset", pred, "-", conf, i + 1)
            
        if model_type == "forest" or (model_type == "auto" and content_info['has_forest']):
            # Detect if forest/vegetation is present
            has_forest, forest_ratio = detect_vegetation(img)
            
            if has_forest:
                forest_class = analyze_forest(img)
                
                # Also analyze vegetation health
                veg_health = analyze_vegetation_health(img)
                
                forest_output = f"{forest_class} (Health: {veg_health['health_category']})"
            else:
                forest_class = "No Vegetation"
                forest_output = forest_class
                
            save_output(OUTPUT_FOREST, img, "Dataset", "-", forest_output, forest_ratio if has_forest else 0.0, i + 1)

    # Display misclassification risk summary
    if misclassification_risks and (model_type == "wildlife" or model_type == "auto"):
        st.markdown("### Wildlife Classification Results")
        
        # Create a table of all results
        results_data = {
            "Image": [os.path.basename(path) for path, _, _, _, _, _ in misclassification_risks],
            "Prediction": [pred for _, pred, _, _, _, _ in misclassification_risks],
            "Original Label": [orig for _, _, orig, _, _, _ in misclassification_risks],
            "Model": [model for _, _, _, model, _, _ in misclassification_risks],
            "Confidence": [f"{conf:.2f}" for _, _, _, _, conf, _ in misclassification_risks],
            "Risk": [risk for _, _, _, _, _, risk in misclassification_risks]
        }
        st.table(results_data)
        
        # Highlight high risk classifications
        high_risk_count = 0
        for path, pred, orig, model, conf, risk in misclassification_risks:
            if risk == "High":
                high_risk_count += 1
                st.warning(f"⚠️ High misclassification risk: {os.path.basename(path)} → {pred}")
        
        if high_risk_count == 0:
            st.success("✅ No high-risk misclassifications detected.")

    st.success(f"✅ {model_type.capitalize()} dataset processed and saved.")

# Add this function to properly combine CNN and federated model predictions
def combine_model_predictions(cnn_class, cnn_conf, fed_class, fed_conf):
    """Intelligently combine predictions from CNN and federated models"""
    
    # If either model is uncertain, use the other model's prediction
    if cnn_class == "Uncertain" and fed_class != "Uncertain":
        print("Using federated model prediction (CNN uncertain)")
        return fed_class, fed_conf, "Federated"
    
    if fed_class == "Uncertain" and cnn_class != "Uncertain":
        print("Using CNN prediction (Federated uncertain)")
        return cnn_class, cnn_conf, "CNN"
    
    # If both models agree on the class, boost confidence
    if cnn_class == fed_class:
        print("Models agree on class, boosting confidence")
        # Weighted average of confidences
        combined_conf = (cnn_conf * 0.6) + (fed_conf * 0.4)
        return cnn_class, combined_conf, "Both"
    
    # If models disagree, use class-specific preference if available
    if cnn_class in st.session_state.class_model_preference:
        preferred_model = st.session_state.class_model_preference[cnn_class]
        if preferred_model == 'cnn':
            print(f"Using CNN for class {cnn_class} (preferred model)")
            return cnn_class, cnn_conf, "CNN"
    
    if fed_class in st.session_state.class_model_preference:
        preferred_model = st.session_state.class_model_preference[fed_class]
        if preferred_model == 'federated':
            print(f"Using Federated for class {fed_class} (preferred model)")
            return fed_class, fed_conf, "Federated"
    
    # Default case: select model with higher confidence
    if cnn_conf >= fed_conf:
        print(f"Using CNN prediction (higher confidence: {cnn_conf:.4f} vs {fed_conf:.4f})")
        return cnn_class, cnn_conf, "CNN"
    else:
        print(f"Using Federated prediction (higher confidence: {fed_conf:.4f} vs {cnn_conf:.4f})")
        return fed_class, fed_conf, "Federated"

# ==== Debug Viewer with Enhanced Analysis ====
def display_debug_view(img):
    """Show debugging information for misclassifications"""
    
    # Tabs for different analysis aspects
    tab1, tab2, tab3, tab4 = st.tabs(["Content Detection", "Model Comparison", "Class Labels", "Raw Predictions"])
    
    with tab1:
        st.write("### Content Detection Analysis")
        
        # Detect content
        content_info = detect_image_content(img)
        
        # Display content detection results
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Forest Detected:** {'Yes' if content_info['has_forest'] else 'No'} ({content_info['forest_confidence']:.2f})")
            
            # Show forest details if present
            if content_info['has_forest']:
                st.markdown("#### Forest Analysis")
                forest_type = analyze_forest(img)
                st.markdown(f"**Forest Type:** {forest_type}")
                
                # Analyze vegetation health
                veg_health = analyze_vegetation_health(img)
                st.markdown(f"**Health:** {veg_health['health_category']} ({veg_health['health_score']}/100)")
                st.markdown(f"**Development Potential:** {veg_health['development_category']} ({veg_health['development_potential']:.1f}/100)")
        
        with col2:
            st.markdown(f"**Wildlife Detected:** {'Yes' if content_info['has_wildlife'] else 'No'} ({content_info['wildlife_confidence']:.2f})")
            
            # Show wildlife details if present
            if content_info['has_wildlife']:
                st.markdown("#### Wildlife Analysis")
                cnn_class, cnn_conf, _, _, _, cnn_orig = classify_wildlife(img)
                st.markdown(f"**CNN Classification:** {cnn_class} ({cnn_conf:.2f})")
                
                if federated_model is not None:
                    fed_class, fed_conf, _, _, _, fed_orig = classify_federated(img)
                    st.markdown(f"**Federated Classification:** {fed_class} ({fed_conf:.2f})")
        
        # Display environmental factors
        st.markdown("#### Environmental Factors")
        has_water, water_ratio = detect_water(img)
        st.markdown(f"**Water Present:** {'Yes' if has_water else 'No'} ({water_ratio:.2f})")
        
        # Display dominant colors
        st.markdown("#### Dominant Colors")
        dom_colors = content_info['dominant_colors']
        color_cols = st.columns(3)
        for i, color in enumerate(dom_colors):
            with color_cols[i]:
                st.markdown(f"**Color {i+1}:** RGB({color[0]}, {color[1]}, {color[2]})")
                st.markdown(f'<div style="background-color:rgb({color[0]},{color[1]},{color[2]}); width:50px; height:50px; border-radius:5px;"></div>', unsafe_allow_html=True)
    
    with tab2:
        st.write("### Direct Model Comparison")
        
        # Apply minimal preprocessing
        img_enhanced = enhance_animal_contrast(img)
        
        # Show preprocessing
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original Input")
        with col2:
            st.image(img_enhanced, caption="Enhanced Input (Used for Classification)")
        
        # Get raw predictions from each model
        cnn_class, cnn_confidence, cnn_second, cnn_second_conf, cnn_uncertainty, cnn_orig = classify_wildlife(img_enhanced)
        
        if federated_model is not None:
            fed_class, fed_confidence, fed_second, fed_second_conf, fed_uncertainty, fed_orig = classify_federated(img_enhanced)
        else:
            fed_class, fed_confidence, fed_second, fed_second_conf, fed_uncertainty, fed_orig = None, 0.0, None, 0.0, 1.0, None

        # Apply calibration
        calibrated_cnn_conf = cnn_confidence * st.session_state.calibration_factors['cnn']
        calibrated_fed_conf = fed_confidence * st.session_state.calibration_factors['federated'] if fed_confidence else 0.0
        
        # Apply class-specific biases if available
        if cnn_class in st.session_state.class_model_preference and st.session_state.class_model_preference[cnn_class] == 'cnn':
            calibrated_cnn_conf *= 1.1
        
        if fed_class in st.session_state.class_model_preference and st.session_state.class_model_preference[fed_class] == 'federated':
            calibrated_fed_conf *= 1.1

        # Show direct comparison table
        comparison_data = {
            "Model": ["CNN", "Federated"],
            "Original Class": [cnn_orig, fed_orig if fed_orig else "N/A"],
            "Mapped Class": [cnn_class, fed_class if fed_class else "N/A"],
            "Raw Confidence": [f"{cnn_confidence:.2f}", f"{fed_confidence:.2f}" if fed_class else "N/A"],
            "Calibrated Confidence": [f"{calibrated_cnn_conf:.2f}", f"{calibrated_fed_conf:.2f}" if fed_class else "N/A"],
            "2nd Choice": [f"{cnn_second} ({cnn_second_conf:.2f})", f"{fed_second} ({fed_second_conf:.2f})" if fed_second else "N/A"]
        }
        st.table(comparison_data)
    
    with tab3:
        st.write("### Class Label Mapping")
        st.info("This shows how original class labels are mapped to standardized ones to reduce misclassifications")
        
        # Display class mapping
        mapping_data = {
            "Original Class": list(class_mapping.keys()),
            "Standardized Class": list(class_mapping.values())
        }
        st.table(mapping_data)
    
    with tab4:
        st.write("### Raw Model Predictions")
        
        # CNN raw predictions
        img_array = np.expand_dims(np.array(img.resize(IMG_SIZE_WILDLIFE)) / 255.0, axis=0)
        cnn_preds = wildlife_model.predict(img_array)[0]
        
        # Sort predictions and get top 5
        cnn_indices = np.argsort(cnn_preds)[-5:][::-1]
        
        # Create CNN predictions table
        cnn_data = {
            "Class": [original_wildlife_classes[i] for i in cnn_indices],
            "Standardized": [class_mapping[original_wildlife_classes[i]] for i in cnn_indices],
            "Confidence": [f"{cnn_preds[i]:.4f}" for i in cnn_indices]
        }
        
        st.write("#### CNN Top 5 Predictions")
        st.table(cnn_data)
        
        # Display bar chart
        fig, ax = plt.subplots(figsize=(10, 4))
        classes = [class_mapping[original_wildlife_classes[i]] for i in cnn_indices]
        values = [cnn_preds[i] for i in cnn_indices]
        ax.bar(classes, values)
        ax.set_title('CNN Top 5 Class Probabilities')
        ax.set_ylabel('Probability')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Federated model raw predictions (if available)
        if federated_model is not None:
            st.write("#### Federated Model Predictions")
            
            # Get predictions
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize(IMG_SIZE_FEDERATED),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                outputs = federated_model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                
                # Get top 5 predictions
                top_confidences, top_indices = torch.topk(probs, 5, dim=1)
                
                # Create table data
                fed_data = {
                    "Class": [original_wildlife_classes[i.item()] for i in top_indices[0]],
                    "Standardized": [class_mapping[original_wildlife_classes[i.item()]] for i in top_indices[0]],
                    "Confidence": [f"{conf.item():.4f}" for conf in top_confidences[0]]
                }
                st.table(fed_data)
                
                # Display bar chart
                fig, ax = plt.subplots(figsize=(10, 4))
                classes = [class_mapping[original_wildlife_classes[i.item()]] for i in top_indices[0]]
                values = [conf.item() for conf in top_confidences[0]]
                ax.bar(classes, values)
                ax.set_title('Federated Model Top 5 Class Probabilities')
                ax.set_ylabel('Probability')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)

def send_poacher_alert(image_path=None):
    """Send email alert when poacher is detected in any image"""
    try:
        # Create email message
        msg = EmailMessage()
        msg['Subject'] = "⚠️ ALERT: Potential Poacher Detected!"
        msg['From'] = GMAIL_SENDER
        msg['To'] = RECEIVER_EMAIL
        
        # Email content
        msg.set_content(f"""
        ⚠️ IMPORTANT: Potential poacher detected in forest monitoring system.
        
        Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Please check the monitoring dashboard immediately.
        """)
        
        # Add image attachment if available
        if image_path:
            with open(image_path, 'rb') as img_file:
                img_data = img_file.read()
                msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename='poacher_detected.jpg')
        
        # Send email with SSL
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(GMAIL_SENDER, GMAIL_PASSWORD)
            server.send_message(msg)
        
        # Show Windows notification
        try:
            toaster = ToastNotifier()
            toaster.show_toast("Forest Monitoring Alert", 
                              "⚠️ Potential poacher detected!", 
                              duration=10,
                              threaded=True)
        except Exception as e:
            print(f"Toast notification error: {e}")
            
        return True
    except Exception as e:
        print(f"Email alert error: {e}")
        return False
# ==== UI ====
def main():
    st.title("🌍 Enhanced Multiple-Model Forest Monitoring System")
    
    # Model status indicators with class counts
    st.markdown(f"### System Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success(f"✅ Wildlife CNN: {len(wildlife_classes)} classes")
    with col2:
        if federated_model is not None:
            st.success(f"✅ Federated Model: {len(wildlife_classes)} classes")
        else:
            st.error("❌ Federated Model Failed")
    with col3:
        st.success("✅ Forest Model Loaded")
    
    # Model calibration controls
    with st.expander("Model Calibration Settings"):
        st.info("Adjust these settings to fine-tune model selection")
        
        model_bias = st.slider("Model Bias (CNN ← → Federated)", 
                             min_value=-0.5, max_value=0.5, value=0.15, step=0.05,
                             help="Positive values favor CNN, negative values favor Federated model")
        
        st.session_state.calibration_factors['cnn'] = 1.0 + model_bias
        st.session_state.calibration_factors['federated'] = 1.0 - model_bias
        
        st.write(f"Current calibration factors: CNN = {st.session_state.calibration_factors['cnn']:.2f}, Federated = {st.session_state.calibration_factors['federated']:.2f}")
        
        # Allow editing class preferences
        st.write("### Class-Specific Model Preferences")
        st.write("Select which model performs better for each class:")
        
        col1, col2 = st.columns(2)
        updated_preferences = {}
        
        # Create a sorted list of unique classes
        unique_classes = sorted(set(class_mapping.values()))
        
        for i, class_name in enumerate(unique_classes):
            # Alternate columns for better layout
            with col1 if i % 2 == 0 else col2:
                current_pref = st.session_state.class_model_preference.get(class_name, 'none')
                options = [('none', 'No Preference'), ('cnn', 'CNN Model'), ('federated', 'Federated Model')]
                selected = st.radio(f"{class_name}", options, 
                                  format_func=lambda x: x[1], 
                                  index=[opt[0] for opt in options].index(current_pref),
                                  key=f"pref_{class_name}")
                updated_preferences[class_name] = selected[0]
        
        # Update preferences if apply button is clicked
        if st.button("Apply Class Preferences"):
            st.session_state.class_model_preference = updated_preferences
            st.success("✅ Class preferences updated")
    
    # Display class standardization info
    with st.expander("View Class Standardization"):
        st.info("To reduce misclassification, original class labels are standardized.")
        
        mapping_data = {
            "Original Class": list(class_mapping.keys()),
            "Standardized Class": list(class_mapping.values())
        }
        st.table(mapping_data)

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Image", "📦 Wildlife CNN", "🌳 Forest CNN", "🔍 Debug Tools"])

    with tab1:
        st.markdown("### Upload Wildlife or Forest Image")
        file = st.file_uploader("Upload image to classify...", type=["jpg", "png", "jpeg"])
        if file:
            img = Image.open(file).convert("RGB")
            
            # Configuration options
            with st.expander("Advanced Configuration"):
                direct_mode = st.checkbox("Use Direct Classification (No Preprocessing)", value=True)
                show_debug = st.checkbox("Show Extended Analysis", value=False)
            
            count = len(list(OUTPUT_REALTIME.glob("*.jpg"))) + 1
            
            # Process image
            process_combined(img, count)
            
            # Show debug view if requested
            if show_debug:
                st.markdown("### Extended Analysis")
                display_debug_view(img)

    with tab2:
        st.markdown("### Wildlife Dataset Classification")
        if st.button("Process Wildlife Dataset Samples"):
            deploy_dataset(WILDLIFE_TRAIN_DIR, model_type="wildlife")

    with tab3:
        st.markdown("### Forest Classification")
        if st.button("Process Forest Dataset Samples"):
            deploy_dataset(FOREST_TRAIN_DIR, model_type="forest")
            
    with tab4:
        st.markdown("### Debugging Tools")
        st.write("Upload an image to analyze classification issues:")
        debug_file = st.file_uploader("Upload image for analysis", type=["jpg", "png"], key="debug_uploader")
        if debug_file:
            debug_img = Image.open(debug_file).convert("RGB")
            st.image(debug_img, caption="Image for Analysis", use_column_width=True)
            display_debug_view(debug_img)

    # Show footer with information
    st.markdown("---")
    st.info("This enhanced system intelligently detects image content and provides detailed analysis of both wildlife and forest vegetation.")

if __name__ == "__main__":
    main()
