import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers, losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt

# Seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Paths
train_dir = r"C:\Gen AI project\wildlife classification\grouped_data\train"
test_dir = r"C:\Gen AI project\wildlife classification\grouped_data\test"

# Hyperparameters
IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 32
EPOCHS = 100

def build_custom_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(512, kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss=losses.CategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    return model

def create_data_augmentation():
    return ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.2
    )

def load_data(datagen):
    train_gen = datagen.flow_from_directory(
        train_dir, target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE, class_mode='categorical',
        subset='training', shuffle=True, seed=42
    )

    val_gen = datagen.flow_from_directory(
        train_dir, target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE, class_mode='categorical',
        subset='validation', shuffle=True, seed=42
    )

    test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        test_dir, target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE, class_mode='categorical',
        shuffle=False
    )

    return train_gen, val_gen, test_gen

def plot_training_metrics(history):
    plt.figure(figsize=(12, 5))
    for i, metric in enumerate(['accuracy', 'loss'], 1):
        plt.subplot(1, 2, i)
        plt.plot(history.history[metric], label=f'Train {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Val {metric}')
        plt.title(f'{metric.capitalize()} over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
    plt.tight_layout()
    plt.savefig("wildlife_training_metrics.png")
    plt.show()

def train_wildlife_classifier():
    datagen = create_data_augmentation()
    train_gen, val_gen, test_gen = load_data(datagen)

    num_classes = len(train_gen.class_indices)
    model = build_custom_cnn_model((IMG_HEIGHT, IMG_WIDTH, 3), num_classes)
    model.summary()

    checkpoint = ModelCheckpoint("best_wildlife_model.keras", monitor='val_accuracy', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[checkpoint, reduce_lr],
        verbose=1
    )

    test_loss, test_acc = model.evaluate(test_gen)
    print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")
    print(f"✅ Test Loss: {test_loss:.4f}")

    model.save("final_wildlife_model.keras")
    plot_training_metrics(history)

    print("\nClass Indices:")
    for class_name, index in train_gen.class_indices.items():
        print(f"{class_name}: {index}")

if __name__ == "__main__":
    train_wildlife_classifier()
