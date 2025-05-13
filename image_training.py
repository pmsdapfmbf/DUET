import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torch.optim.lr_scheduler import StepLR
import numpy as np

class AdvancedCNN(nn.Module):
    def __init__(self, num_classes):
        super(AdvancedCNN, self).__init__()
        
        # First Conv Block: 1 -> 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Second Conv Block: 64 -> 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Third Conv Block: 128 -> 256
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 7 * 7, 512)  # Adjust dimensions based on input image size
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Max Pooling and Dropout for regularization
        self.pool = nn.MaxPool2d(2, 2)  # Pooling with 2x2 window
        self.dropout = nn.Dropout(0.5)  # Dropout with probability 0.5
    
    def forward(self, x):
        # First Conv Block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn1(self.conv2(x)))
        x = self.pool(x)
        
        # Second Conv Block
        x = F.relu(self.bn2(self.conv3(x)))
        x = F.relu(self.bn2(self.conv4(x)))
        x = self.pool(x)
        
        # Third Conv Block
        x = F.relu(self.bn3(self.conv5(x)))
        x = F.relu(self.bn3(self.conv6(x)))
        x = self.pool(x)
        
        # Flatten before fully connected layers
        x = x.view(-1, 256 * 7 * 7)  # Adjust dimensions based on input size
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout to avoid overfitting
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

def get_vgg_model():
    vgg = models.vgg16_bn(weights="IMAGENET1K_V1")
    vgg.features[0] = nn.Conv2d(in_channels=1, 
                                out_channels=64, 
                                kernel_size=3, 
                                stride=1, 
                                padding=1)

    # Copy the weights from the first layer of the original VGG16 model.
    # We'll average the RGB weights for each of the 3 input channels to convert them to a single grayscale channel.
    with torch.no_grad():
        vgg.features[0].weight = nn.Parameter(torch.mean(vgg.features[0].weight, dim=1, keepdim=True))

    for param in vgg.parameters():
        param.requires_grad = False
    
    # old implementation
    # vgg.classifier[6].out_features = 2

    # replace to 2
    num_features = vgg.classifier[6].in_features  # Get input features of last layer (4096)
    vgg.classifier[6] = nn.Linear(num_features, 2)
    
    # Make sure classifier layers require gradient
    for idx, param in enumerate(vgg.classifier.parameters()):
        param.requires_grad = True
    
    torch.nn.init.xavier_uniform_(vgg.classifier[6].weight)
    vgg.classifier[6].bias.data.fill_(0)
    return vgg

def unfreeze_only_last_layer(vgg):
    for param in vgg.parameters():
        param.requires_grad = False
    # set last layer only to true grad

    for name, param in list(vgg.classifier.named_parameters())[-1:]:  # Exclude the last layer
        param.requires_grad = True
    
    return vgg

# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)  # Get the index of the max log-probability
    correct = (preds == labels).sum().item()  # Compare predictions with ground truth
    accuracy = correct / labels.size(0)       # Compute accuracy for the batch
    return accuracy

# Function to evaluate the model on validation set
def evaluate_model(model, dataloader, criterion, device):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    total_accuracy = 0.0
    
    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Update running loss
            running_loss += loss.item()
            
            # Calculate accuracy for the current batch
            accuracy = calculate_accuracy(outputs, labels)
            total_accuracy += accuracy
    
    avg_loss = running_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader) * 100
    return avg_loss, avg_accuracy

# Function to train the model
def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device, seed, printout=True):
    torch.manual_seed(seed)
    model.to(device)
    all_acc_validation = []
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)  # Reduce lr by 0.1 every 10 epochs
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        total_accuracy = 0.0
        batch_idx = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update running loss
            running_loss += loss.item()
            
            # Calculate accuracy for the current batch
            accuracy = calculate_accuracy(outputs, labels)
            total_accuracy += accuracy
            batch_idx+=1
        
        # Average training loss and accuracy
        avg_train_loss = running_loss / len(train_loader)
        avg_train_accuracy = total_accuracy / len(train_loader) * 100
        
        # Validation phase
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        all_acc_validation.append(val_accuracy)
        # Print statistics for each epoch
        if printout:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    return all_acc_validation, val_accuracy, model

def train(train_loader, validation_loader, seed, num_epochs=10, cuda="cuda:0", printout=True, lr=1e-5, num_layer_to_unfreeze=1):

    # Initialize model, loss function, and optimizer
    model = get_vgg_model()
    criterion = nn.CrossEntropyLoss()  # For classification tasks
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)

    for name, param in list(model.classifier.named_parameters()):  # Exclude the last layer
        param.requires_grad = False
      
    for name, param in list(model.classifier.named_parameters())[-2 * num_layer_to_unfreeze:]:  # Exclude the last layer
        param.requires_grad = True

        
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    # Train the model
    all_acc, final_acc, model = train_model(model, train_loader, validation_loader, num_epochs, criterion, optimizer, device, printout=printout, seed=seed)
    return all_acc, final_acc, model

def run_trials(train_loader, validation_loader, cuda, epochs, num_layer_to_unfreeze=2, trials=10, printout=True):
    acc_obs = []
    for x in range(trials):
        all_acc,final_acc,_ = train(train_loader, validation_loader, seed=x, num_epochs=epochs, printout=printout, lr=5e-5, cuda=cuda, num_layer_to_unfreeze=num_layer_to_unfreeze)
        acc_obs.append(max(all_acc))
    print("avg acc across trials: ", np.mean(acc_obs))
    print("avg std across trials: ", np.std(acc_obs))
    return acc_obs