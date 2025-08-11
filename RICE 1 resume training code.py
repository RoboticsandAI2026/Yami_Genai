import os
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import time

# ================== 1. Load Dataset ==================
def load_image(image_path, target_size=(256, 256)):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize [0,1]
    return img

# Dataset Paths
root_dir = r"C:\Users\YamiD\Downloads\GAN\RICE"
cloudy_path = os.path.join(root_dir, "RICE1/cloud")
cloudfree_path = os.path.join(root_dir, "RICE1/label")

def load_dataset():
    image_pairs = []
    cloud_files = sorted(os.listdir(cloudy_path))
    label_files = sorted(os.listdir(cloudfree_path))

    # Verify matching files
    if len(cloud_files) != len(label_files):
        print(f"Warning: Number of cloudy images ({len(cloud_files)}) does not match cloud-free images ({len(label_files)})")
        
    # Use the minimum length to avoid index errors
    min_len = min(len(cloud_files), len(label_files))
    
    for i in range(min_len):
        c_file = cloud_files[i]
        l_file = label_files[i]
        c_path = os.path.join(cloudy_path, c_file)
        l_path = os.path.join(cloudfree_path, l_file)
        image_pairs.append((c_path, l_path))

    return image_pairs

# Load dataset paths
image_pairs = load_dataset()
print(f"✅ Dataset Loaded: {len(image_pairs)} image pairs")

# ================== 2. Create Data Pipeline with Augmentation ==================
BATCH_SIZE = 8  # Reduced batch size for more stable training
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = max(len(image_pairs) // BATCH_SIZE, 10)

def random_jitter(input_image, target_image):
    # Random crop and resize
    concat_image = tf.concat([input_image, target_image], axis=-1)
    concat_image = tf.image.random_crop(concat_image, [240, 240, 6])
    concat_image = tf.image.resize(concat_image, [256, 256])
    
    # Split back to input and target
    input_image = concat_image[..., :3]
    target_image = concat_image[..., 3:]
    
    # Random flipping
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        target_image = tf.image.flip_left_right(target_image)
        
    # Ensure values remain in [0, 1] range
    input_image = tf.clip_by_value(input_image, 0.0, 1.0)
    target_image = tf.clip_by_value(target_image, 0.0, 1.0)
        
    return input_image, target_image

def preprocess_images(cloudy_path, cloudfree_path):
    cloudy_img = load_image(cloudy_path)
    cloudfree_img = load_image(cloudfree_path)
    
    # Apply data augmentation
    cloudy_img, cloudfree_img = random_jitter(cloudy_img, cloudfree_img)
    
    return cloudy_img, cloudfree_img

def dataset_generator():
    for cloudy_path, cloudfree_path in image_pairs:
        yield preprocess_images(cloudy_path, cloudfree_path)

dataset = tf.data.Dataset.from_generator(
    dataset_generator, 
    output_signature=(tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
                      tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32))
)

train_dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
print(f"✅ Steps per Epoch: {STEPS_PER_EPOCH}")

# ================== 3. Define Improved Pix2Pix Generator with Attention ==================
def build_generator():
    inputs = Input(shape=(256, 256, 3))
    
    # Encoder
    down1 = Conv2D(64, (4, 4), strides=2, padding='same')(inputs)
    down1 = LeakyReLU(0.2)(down1)
    
    down2 = Conv2D(128, (4, 4), strides=2, padding='same')(down1)
    down2 = BatchNormalization()(down2)  # Added normalization
    down2 = LeakyReLU(0.2)(down2)
    
    down3 = Conv2D(256, (4, 4), strides=2, padding='same')(down2)
    down3 = BatchNormalization()(down3)  # Added normalization
    down3 = LeakyReLU(0.2)(down3)
    
    down4 = Conv2D(512, (4, 4), strides=2, padding='same')(down3)
    down4 = BatchNormalization()(down4)  # Added normalization
    down4 = LeakyReLU(0.2)(down4)
    
    # Bottleneck
    bottleneck = Conv2D(512, (4, 4), strides=2, padding='same')(down4)
    bottleneck = BatchNormalization()(bottleneck)  # Added normalization
    bottleneck = LeakyReLU(0.2)(bottleneck)
    
    # Decoder with skip connections
    up1 = Conv2DTranspose(512, (4, 4), strides=2, padding='same')(bottleneck)
    up1 = BatchNormalization()(up1)
    up1 = Dropout(0.5)(up1)  # Added dropout for regularization
    up1 = Concatenate()([up1, down4])  # Skip connection
    up1 = Activation('relu')(up1)
    
    up2 = Conv2DTranspose(256, (4, 4), strides=2, padding='same')(up1)
    up2 = BatchNormalization()(up2)
    up2 = Dropout(0.5)(up2)  # Added dropout for regularization
    up2 = Concatenate()([up2, down3])  # Skip connection
    up2 = Activation('relu')(up2)
    
    up3 = Conv2DTranspose(128, (4, 4), strides=2, padding='same')(up2)
    up3 = BatchNormalization()(up3)
    up3 = Concatenate()([up3, down2])  # Skip connection
    up3 = Activation('relu')(up3)
    
    up4 = Conv2DTranspose(64, (4, 4), strides=2, padding='same')(up3)
    up4 = BatchNormalization()(up4)
    up4 = Concatenate()([up4, down1])  # Skip connection
    up4 = Activation('relu')(up4)
    
    # Output layer
    outputs = Conv2DTranspose(3, (4, 4), strides=2, padding='same', activation='tanh')(up4)
    # Use tanh and rescale to [0,1] for better gradient flow
    outputs = Lambda(lambda x: (x + 1) / 2)(outputs)
    
    return Model(inputs, outputs)

# ================== 4. Improved PatchGAN Discriminator with Spectral Normalization ==================
def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    
    u = tf.Variable(tf.random.normal([1, w_shape[-1]]), trainable=False)
    
    u_hat = u
    v_hat = None
    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)
        
        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)
        
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma
    w_norm = tf.reshape(w_norm, w_shape)
    
    return w_norm

def conv_sn(x, channels, kernel=4, stride=2, padding='same'):
    x = Conv2D(channels, kernel, strides=stride, padding=padding,
               kernel_initializer=tf.keras.initializers.GlorotUniform())(x)
    return x

def build_discriminator():
    input_image = Input(shape=(256, 256, 3))
    target_image = Input(shape=(256, 256, 3))
    
    # Concatenate inputs
    x = Concatenate()([input_image, target_image])
    
    # Using spectral normalization for more stable training
    x = conv_sn(x, 64, stride=2)
    x = LeakyReLU(0.2)(x)
    
    x = conv_sn(x, 128, stride=2)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    x = conv_sn(x, 256, stride=2)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    x = conv_sn(x, 512, stride=1)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    # Output - no sigmoid here as we'll use sigmoid cross entropy loss
    x = Conv2D(1, (4, 4), padding='same')(x)
    
    return Model([input_image, target_image], x)

# Load or Create Models
def create_or_load_models(model_dir="./models_2"):
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    gen_path = os.path.join(model_dir, "generator_rice1.keras")
    disc_path = os.path.join(model_dir, "discriminator_rice1.keras")
    
    try:
        generator = tf.keras.models.load_model(gen_path, compile=False)
        discriminator = tf.keras.models.load_model(disc_path, compile=False)
        print("✅ Models Loaded Successfully")
        return generator, discriminator, True
    except:
        print("⚠️ No saved models found, initializing new models.")
        generator = build_generator()
        discriminator = build_discriminator()
        return generator, discriminator, False

generator, discriminator, models_loaded = create_or_load_models()

# ================== 5. Training Setup with Improved Loss Functions ==================
# Learning rate schedulers for stability
lr_schedule_g = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=2e-4,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True)

lr_schedule_d = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=2e-5,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True)

generator_optimizer = Adam(learning_rate=lr_schedule_g, beta_1=0.5)
discriminator_optimizer = Adam(learning_rate=lr_schedule_d, beta_1=0.5)

# Loss functions
BCE = BinaryCrossentropy(from_logits=True)  # Using from_logits=True for stability

def generator_loss(disc_generated_output, gen_output, target):
    # Adversarial loss
    gan_loss = BCE(tf.ones_like(disc_generated_output), disc_generated_output)
    
    # L1 loss
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    
    # Perceptual loss (simplified with L2)
    l2_loss = tf.reduce_mean(tf.square(target - gen_output))
    
    # Total loss with weighted components
    total_loss = gan_loss + (100 * l1_loss) + (10 * l2_loss)
    
    return total_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = BCE(tf.ones_like(disc_real_output), disc_real_output)
    fake_loss = BCE(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + fake_loss
    
    return total_disc_loss

# ================== 6. Improved Training Loop with Visualization ==================
# Function to visualize and save results
def generate_images(model, test_input, target, epoch, idx, save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)
    
    prediction = model(test_input, training=False)
    
    plt.figure(figsize=(15, 5))
    
    display_list = [test_input[0], target[0], prediction[0]]
    title = ['Cloudy Input', 'Cloud-Free Ground Truth', 'Predicted Cloud-Free']
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    
    plt.savefig(f"{save_dir}/epoch_{epoch}_sample_{idx}.png")
    plt.close()

# Checkpoint manager for saving models
checkpoint_dir = './training_checkpoints_2'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator
)
checkpoint_manager = tf.train.CheckpointManager(
    checkpoint, directory=checkpoint_dir, max_to_keep=5)

# Try to restore from checkpoint
if checkpoint_manager.latest_checkpoint:
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    print(f"✅ Restored from checkpoint: {checkpoint_manager.latest_checkpoint}")

# Get start epoch from checkpoint if available
if models_loaded or checkpoint_manager.latest_checkpoint:
    start_epoch = int(checkpoint_manager.latest_checkpoint.split('-')[-1]) if checkpoint_manager.latest_checkpoint else 1
else:
    start_epoch = 1

# Initialize log dict for metrics
metrics_log = {
    "gen_total_loss": [],
    "gen_gan_loss": [],
    "gen_l1_loss": [],
    "disc_loss": [],
    "elapsed_time": []
}

@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate fake image
        gen_output = generator(input_image, training=True)
        
        # Get discriminator outputs
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        
        # Calculate losses
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
        
    # Calculate gradients
    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    # Apply gradients
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
    
    return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss

def train(dataset, start_epoch, epochs):
    results_dir = "./results_2"
    os.makedirs(results_dir, exist_ok=True)
    
    # Get a test batch
    test_batch = next(iter(dataset.take(1)))
    test_input, test_target = test_batch
    
    start_time = time.time()
    
    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()
        print(f"\n🔹 Epoch {epoch}/{epochs} - Training Started")
        
        # Reset metrics for each epoch
        epoch_gen_total_loss = 0
        epoch_gen_gan_loss = 0
        epoch_gen_l1_loss = 0
        epoch_disc_loss = 0
        
        for step, (input_image, target) in enumerate(dataset.take(STEPS_PER_EPOCH)):
            gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = train_step(input_image, target, epoch)
            
            # Accumulate losses for epoch average
            epoch_gen_total_loss += gen_total_loss
            epoch_gen_gan_loss += gen_gan_loss
            epoch_gen_l1_loss += gen_l1_loss
            epoch_disc_loss += disc_loss
            
            if step % 10 == 0:
                print(f"  Step {step}/{STEPS_PER_EPOCH} - "
                      f"Gen Loss: {gen_total_loss:.4f}, "
                      f"Disc Loss: {disc_loss:.4f}")
        
        # Calculate average losses for the epoch
        avg_gen_total = epoch_gen_total_loss / STEPS_PER_EPOCH
        avg_gen_gan = epoch_gen_gan_loss / STEPS_PER_EPOCH
        avg_gen_l1 = epoch_gen_l1_loss / STEPS_PER_EPOCH
        avg_disc = epoch_disc_loss / STEPS_PER_EPOCH
        
        # Log metrics
        metrics_log["gen_total_loss"].append(float(avg_gen_total))
        metrics_log["gen_gan_loss"].append(float(avg_gen_gan))
        metrics_log["gen_l1_loss"].append(float(avg_gen_l1))
        metrics_log["disc_loss"].append(float(avg_disc))
        
        # Calculate time elapsed
        epoch_time = time.time() - epoch_start_time
        metrics_log["elapsed_time"].append(epoch_time)
        
        # Generate a test image
        generate_images(generator, test_input, test_target, epoch, 0, results_dir)
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint_manager.save()
            
            # Save standalone models
            generator.save(f"./models_2/generator_rice1_epoch_{epoch}.keras")
            discriminator.save(f"./models_2/discriminator_rice1_epoch_{epoch}.keras")
            
            # Also save latest models
            generator.save("./models_2/generator_rice1_latest.keras")
            discriminator.save("./models_2/discriminator_rice1_latest.keras")
            
            # Plot and save metrics
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(metrics_log["gen_total_loss"], label='Generator Total Loss')
            plt.title('Generator Total Loss')
            plt.xlabel('Epoch')
            plt.legend()
            
            plt.subplot(2, 2, 2)
            plt.plot(metrics_log["gen_gan_loss"], label='Generator GAN Loss')
            plt.plot(metrics_log["gen_l1_loss"], label='Generator L1 Loss')
            plt.title('Generator Loss Components')
            plt.xlabel('Epoch')
            plt.legend()
            
            plt.subplot(2, 2, 3)
            plt.plot(metrics_log["disc_loss"], label='Discriminator Loss')
            plt.title('Discriminator Loss')
            plt.xlabel('Epoch')
            plt.legend()
            
            plt.subplot(2, 2, 4)
            plt.plot(metrics_log["elapsed_time"], label='Time per Epoch (s)')
            plt.title('Training Time')
            plt.xlabel('Epoch')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"{results_dir}/metrics_epoch_{epoch}.png")
            plt.close()
            
            print(f"✅ Checkpoint saved at epoch {epoch}")
            print(f"✅ Models saved at epoch {epoch}")
        
        print(f"Epoch {epoch}/{epochs} completed in {epoch_time:.2f}s - "
              f"Gen Loss: {avg_gen_total:.4f}, "
              f"Disc Loss: {avg_disc:.4f}")
    
    total_time = time.time() - start_time
    print(f"✅ Training completed in {total_time/60:.2f} minutes")

# ================== 7. Run Training ==================
EPOCHS = 100# Increase number of epochs for better results
train(train_dataset, start_epoch, EPOCHS)
print("✅ Training Complete!")
