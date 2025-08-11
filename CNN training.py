import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ✅ Set paths to train and test directories
train_dir = r"C:\Gen AI project\forest change classification\split_data\train"
test_dir = r"C:\Gen AI project\forest change classification\split_data\test"

# ✅ Image Parameters
IMG_HEIGHT, IMG_WIDTH = 256, 256  # Image size
BATCH_SIZE = 32  # Number of images per batch
EPOCHS = 50  # Number of training iterations

# ✅ Load Images from Train and Test Directories
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    rotation_range=30,  # Data augmentation
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale for test data

# ✅ Use "input" class_mode to avoid label issues
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='input',  # Autoencoder-style training (output = input)
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='input',  # No labels needed
    shuffle=False
)

# ✅ Define CNN Autoencoder Model
def build_forest_autoencoder():
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Decoder
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)

    outputs = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    model = models.Model(inputs, outputs)
    
    model.compile(optimizer='adam',
                  loss='mse',  # Mean Squared Error for image reconstruction
                  metrics=['accuracy'])

    return model

# ✅ Train the CNN Model
def train_forest_autoencoder():
    model = build_forest_autoencoder()

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=test_generator
    )

    # ✅ Save Model in .keras format
    model.save("forest_conservation_autoencoder.keras")
    
    # ✅ Print Accuracy
    loss, accuracy = model.evaluate(test_generator)
    print(f"\n✅ Final Validation Accuracy: {accuracy * 100:.2f}%")

    return model, history

# ✅ Run Training
if __name__ == "__main__":
    trained_model, history = train_forest_autoencoder()
