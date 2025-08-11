import os
import copy
import time
import random
import argparse
import numpy as np
from PIL import Image
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Configuration parameters
NUM_CLIENTS = 2
BATCH_SIZE = 32
EPOCHS = 5  # Local epochs per round
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CHANNELS = 3  # RGB
NUM_ROUNDS = 25  # Number of federated learning rounds
NUM_CLASSES = 12  # 12 wildlife classes

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class WildlifeDataset(Dataset):
    """Wildlife classification dataset."""
    
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images: Numpy array of images
            labels: Numpy array of labels
            transform: Optional transform to be applied on a sample
        """
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to PIL Image if needed for transforms
        if self.transform:
            image = Image.fromarray((image * 255).astype(np.uint8))
            image = self.transform(image)
        else:
            # Convert to PyTorch tensor
            image = torch.from_numpy(image).float()
            if image.shape[2] == 3:  # If channels are last
                image = image.permute(2, 0, 1)  # Change to channels first (C, H, W)
        
        # Convert label to long tensor (this fixes the error)
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label

def preprocess_image(image_path):
    """Load and preprocess an image for the CNN model."""
    try:
        img = Image.open(image_path).convert('RGB')  # Ensure RGB
        img = img.resize((IMG_HEIGHT, IMG_WIDTH))
        img_array = np.array(img) / 255.0  # Normalize to [0,1]
        return img_array
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        # Return a blank image instead of crashing
        return np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))

def load_dataset(data_dir, client_id=None):
    """
    Load and preprocess the wildlife dataset.
    
    Args:
        data_dir: Root directory of the dataset
        client_id: If specified, only load data for this client
    
    Returns:
        Tuple of (x_train, y_train), (x_test, y_test)
    """
    # If client_id is specified, load only that client's data folder
    if client_id is not None:
        client_dirs = [os.path.join(data_dir, f'client_{client_id}')]
    else:
        # Otherwise, load all client directories
        client_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) 
                      if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('client_')]
    
    images = []
    labels = []
    
    # Get all possible class names across all clients to build a global class index mapping
    all_class_names = set()
    for client_dir in client_dirs:
        class_names = [d for d in os.listdir(client_dir) 
                      if os.path.isdir(os.path.join(client_dir, d))]
        all_class_names.update(class_names)
    
    # Create a mapping from class name to index
    class_to_idx = {name: idx for idx, name in enumerate(sorted(all_class_names))}
    print(f"Global class mapping: {class_to_idx}")
    
    # Assuming each client directory has class subdirectories
    for client_dir in client_dirs:
        class_dirs = [os.path.join(client_dir, d) for d in os.listdir(client_dir) 
                     if os.path.isdir(os.path.join(client_dir, d))]
        
        for class_dir in class_dirs:
            # Get class name from directory name
            class_name = os.path.basename(class_dir)
            class_idx = class_to_idx[class_name]
            
            image_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                          if f.lower().endswith(('png', 'jpg', 'jpeg'))]
            
            print(f"Loading {len(image_files)} images from {class_dir}")
            
            for img_path in image_files:
                try:
                    img_array = preprocess_image(img_path)
                    images.append(img_array)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    if not images:
        raise ValueError(f"No images found in {client_dirs}")
    
    # Convert to numpy arrays
    x_data = np.array(images)
    y_data = np.array(labels)
    
    # Split into train and test (80/20 split)
    indices = np.random.permutation(len(x_data))
    split_idx = int(0.8 * len(indices))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    x_train, y_train = x_data[train_indices], y_data[train_indices]
    x_test, y_test = x_data[test_indices], y_data[test_indices]
    
    return (x_train, y_train), (x_test, y_test)

def create_dataloaders(x_train, y_train, x_test, y_test, batch_size=32):
    """Create PyTorch DataLoaders from numpy arrays."""
    # Create datasets
    train_dataset = WildlifeDataset(x_train, y_train)
    test_dataset = WildlifeDataset(x_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

class WildlifeCNN(nn.Module):
    """CNN model for wildlife classification."""
    
    def __init__(self, num_classes=12):
        super(WildlifeCNN, self).__init__()
        
        # First convolutional block
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
        
        # Calculate input size for first fully connected layer
        fc_input_size = 512 * (IMG_HEIGHT // 16) * (IMG_WIDTH // 16)
        
        # Use a smaller model for memory efficiency
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

def train_model(model, train_loader, epochs=5, lr=0.001):
    """Train a model on a client's data."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
    return model

def test_model(model, test_loader):
    """Test a model on a client's test data."""
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def average_models(global_model, client_models, client_weights):
    """Average the model parameters of multiple clients."""
    global_dict = global_model.state_dict()
    
    for k in global_dict.keys():
        # For batch norm running stats, don't average
        if 'num_batches_tracked' in k:
            continue
            
        # Create a new tensor with the correct dtype
        dtype = global_dict[k].dtype
        
        # For integer types (like Long), we need special handling
        if dtype == torch.long or dtype == torch.int:
            # For integer params, take the value from the most weighted client
            max_weight_idx = np.argmax(client_weights)
            global_dict[k] = client_models[max_weight_idx].state_dict()[k].clone()
        else:
            # For float params, we can do weighted averaging
            global_dict[k] = torch.zeros_like(global_dict[k])
            for client_idx, client_model in enumerate(client_models):
                weight = client_weights[client_idx]
                global_dict[k] += client_model.state_dict()[k] * weight
    
    global_model.load_state_dict(global_dict)
    return global_model

def create_client_data(data_dir):
    """Create a dictionary of client data for federated learning simulation."""
    client_data = {}
    client_weights = []
    total_samples = 0
    
    for client_id in range(1, NUM_CLIENTS + 1):
        # Load data for this client
        (x_train, y_train), (x_test, y_test) = load_dataset(data_dir, client_id)
        
        # Create DataLoaders
        train_loader, test_loader = create_dataloaders(x_train, y_train, x_test, y_test, BATCH_SIZE)
        
        client_data[client_id] = {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'num_train_examples': len(x_train),
            'num_test_examples': len(x_test)
        }
        
        total_samples += len(x_train)
        client_weights.append(len(x_train))
        
        print(f"Client {client_id}: {len(x_train)} training examples, {len(x_test)} test examples")
    
    # Normalize client weights
    client_weights = [w / total_samples for w in client_weights]
    
    return client_data, client_weights

def save_model(model, save_path):
    """Save the PyTorch model to disk."""
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def run_federated_learning(data_dir, output_dir):
    """
    Main function to run federated learning for wildlife classification.
    
    Args:
        data_dir: Directory containing the wildlife image dataset
        output_dir: Directory to save models and results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Log file for tracking metrics
    log_file = open(os.path.join(output_dir, 'federated_metrics.txt'), 'w')
    log_file.write("Round,Accuracy\n")
    
    # Load client data
    print("Loading client data...")
    client_data, client_weights = create_client_data(data_dir)
    
    # Initialize global model
    global_model = WildlifeCNN(num_classes=NUM_CLASSES).to(device)
    
    # Use smaller image size if memory issues occur
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Track metrics across rounds
    metrics = {
        'round_num': [],
        'global_accuracy': []
    }
    
    # Perform federated training
    print(f"Starting federated training for {NUM_ROUNDS} rounds...")
    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n===== Round {round_num}/{NUM_ROUNDS} =====")
        
        # Initialize list to hold client models for this round
        client_models = []
        
        # Train a model for each client
        for client_id, data in client_data.items():
            print(f"\nTraining client {client_id} model...")
            
            # Initialize with current global model weights
            client_model = copy.deepcopy(global_model)
            
            # Train on client data
            client_model = train_model(
                model=client_model,
                train_loader=data['train_loader'],
                epochs=EPOCHS,
                lr=0.001
            )
            
            # Add to list of client models
            client_models.append(client_model)
            
            # Evaluate client model on its test data
            client_accuracy = test_model(client_model, data['test_loader'])
            print(f"Client {client_id} test accuracy: {client_accuracy:.2f}%")
        
        # Average client models to create new global model
        global_model = average_models(global_model, client_models, client_weights)
        
        # Evaluate global model on each client's test data
        print("\nEvaluating global model on all clients...")
        total_correct = 0
        total_samples = 0
        
        for client_id, data in client_data.items():
            client_accuracy = test_model(global_model, data['test_loader'])
            print(f"Client {client_id} test accuracy: {client_accuracy:.2f}%")
            
            # Calculate actual numbers for weighted average
            num_test = client_data[client_id]['num_test_examples']
            correct = int(client_accuracy * num_test / 100)
            
            total_correct += correct
            total_samples += num_test
        
        # Calculate global accuracy
        global_accuracy = 100 * total_correct / total_samples
        print(f"Round {round_num} global test accuracy: {global_accuracy:.2f}%")
        
        # Track metrics
        metrics['round_num'].append(round_num)
        metrics['global_accuracy'].append(global_accuracy)
        
        # Write to log file
        log_file.write(f"{round_num},{global_accuracy:.4f}\n")
        log_file.flush()
        
        # Save model every few rounds or at the end
        if round_num % 5 == 0 or round_num == NUM_ROUNDS:
            save_model_path = os.path.join(output_dir, f'wildlife_model_round_{round_num}.pth')
            save_model(global_model, save_model_path)
    
    # Save the final model
    final_model_path = os.path.join(output_dir, 'wildlife_model_final.pth')
    save_model(global_model, final_model_path)
    
    # Close log file
    log_file.close()
    
    print("\n===== Training Summary =====")
    print(f"Trained for {NUM_ROUNDS} rounds")
    print(f"Final global accuracy: {metrics['global_accuracy'][-1]:.2f}%")
    print(f"Models saved to: {output_dir}")
    
    return metrics

if __name__ == "__main__":
    # Default directory paths
    DATA_DIR =r"C:\Gen AI project\wildlife classification\federated_clients"
    OUTPUT_DIR = r"C:\Gen AI project\wildlife classification\federated_model"
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PyTorch Federated Learning for Wildlife Classification')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Path to save output models and metrics')
    parser.add_argument('--num_rounds', type=int, default=NUM_ROUNDS, help='Number of federated learning rounds')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of local epochs per round')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=None, help='Override image size (use smaller value if out of memory)')
    
    args = parser.parse_args()
    
    # Update configuration with parsed arguments
    DATA_DIR = args.data_dir
    OUTPUT_DIR = args.output_dir
    NUM_ROUNDS = args.num_rounds
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    
    # Optionally update image size
    if args.img_size is not None:
        IMG_HEIGHT = args.img_size
        IMG_WIDTH = args.img_size
        print(f"Using custom image size: {IMG_HEIGHT}x{IMG_WIDTH}")
    
    print(f"Starting federated learning with {NUM_CLIENTS} clients, {NUM_ROUNDS} rounds")
    print(f"Dataset directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Run federated learning
    metrics = run_federated_learning(DATA_DIR, OUTPUT_DIR)
    
    print("Federated learning completed successfully!")
