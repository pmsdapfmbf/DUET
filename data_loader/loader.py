from torch.utils.data import Dataset, DataLoader, Subset, random_split, TensorDataset, ConcatDataset
from PIL import Image
from torchvision import transforms, datasets
from pathlib import Path
from typing import List
import pandas as pd
import struct
import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
import random

class CustomMNISTDataset(Dataset):
    def load_labels(self, label_file):
        with open(label_file, 'rb') as f:
            # Read the magic number and number of labels
            magic, num_labels = struct.unpack('>II', f.read(8))
            # Read the label data
            label_data = np.fromfile(f, dtype=np.uint8)
        return label_data
    
    def load_images(self, image_file):
        with open(image_file, 'rb') as f:
            # Read the magic number and metadata (numbers of images, rows, columns)
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            # Read the image data and reshape it to (num_images, rows, cols)
            image_data = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)
        return image_data

    def __init__(self, image_dir, labels_dir, transform=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            labels_file (string): Path to the labels file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image = self.load_images(image_dir) 
        self.labels = self.load_labels(labels_dir) 
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.image[idx]
        label = self.labels[idx]
        
        # Apply any transformations (e.g., ToTensor)
        if self.transform:
            image = self.transform(image)
        
        return image, label

def filter_dataset_by_labels(dataset, desired_labels):
    # Get all the indices where the labels match the desired ones
    indices = [i for i, (image, label) in enumerate(dataset) if label in desired_labels]
    
    # Use Subset to create a filtered dataset with the matching indices
    filtered_dataset = Subset(dataset, indices)
    
    return filtered_dataset

def get_fashion_mnist(label_filter=[1,0], batch_size=128, portion_training=1.0, seed=0, return_whole=False):
    torch.manual_seed(seed)
    class CustomDataset(Dataset):
        def __init__(self, original_dataset):
            self.original_dataset = original_dataset

        def __getitem__(self, index):
            data, label = self.original_dataset[index]
            label = self.convert_labels(label)
            return data, label

        def __len__(self):
            return len(self.original_dataset)

        def convert_labels(self, label):
            if label == label_filter[0]:
                return 1
            else:
                return 0

    # Define the path to your images and labels
    image_dir = './data/fashion_mnist/train-images'  # Folder where images are stored
    labels_dir = './data/fashion_mnist/train-labels'  # CSV file with image names and labels

    # Define transformations (e.g., convert image to tensor)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(299, antialias=True)])

    # Create the dataset
    mnist_dataset = CustomMNISTDataset(image_dir=image_dir, labels_dir=labels_dir, transform=transform)
    mnist_dataset = filter_dataset_by_labels(mnist_dataset, label_filter) # only want 0 or 1
    
    if not return_whole:
        train_size = int(portion_training * len(mnist_dataset))  # 80% for training
        val_size = len(mnist_dataset) - train_size  # Remaining for validation

        # Randomly split the dataset into training and validation sets
        train_dataset, val_dataset = random_split(mnist_dataset, [train_size, val_size])
    else:
        train_dataset = mnist_dataset
        val_dataset = mnist_dataset
    
    # Create DataLoaders for both datasets
    mnist_train_loader = DataLoader(CustomDataset(train_dataset), batch_size=batch_size, shuffle=True)
    mnist_val_loader = DataLoader(CustomDataset(val_dataset), batch_size=batch_size, shuffle=False)
    
    return mnist_train_loader, mnist_val_loader

def get_cat_dog(batch_size=128, portion_training=1.0, seed=0, return_whole=False):
    torch.manual_seed(seed)
    # Define the directory path where the images are located
    image_dir = 'data/cats_dogs_large'

    transform = transforms.Compose([
        transforms.Resize(299, antialias=True),  # Resize to 299x299
        transforms.Grayscale(1),
        transforms.CenterCrop(299),  # Center crop
        #transforms.RandomRotation(45),
        transforms.GaussianBlur(3),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    # Create a dataset using ImageFolder
    dataset = datasets.ImageFolder(root=image_dir, transform=transform)
    train_size = int(portion_training * len(dataset))  # 80% for training
    val_size = len(dataset) -  train_size # Remaining for validation

    # Randomly split the dataset into training and validation sets
    if not return_whole:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    else:
        train_dataset = dataset
        val_dataset = dataset
    # train_size = int(0.5 * len(val_dataset))  # 80% for training
    # val_size = len(val_dataset) -  train_size # Remaining for validation
    #train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # Create DataLoaders for both datasets
    cat_dog_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    cat_dog_val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Check the class-to-index mapping (class labels)
    print(f"Class to index mapping: {dataset.class_to_idx}")

    # Iterate over the DataLoader
    for images, labels in cat_dog_val_loader:
        print(f"Batch of images: {images.shape}, Batch of labels: {labels}")
        break  # Just to show the first batch
    return cat_dog_train_loader, cat_dog_val_loader

def sample_from_dataloader(original_data_loader, seed, num_batches=3, contaminate=False) -> TensorDataset:
    torch.manual_seed(seed)
    sampled_data = []

    # Sample from the original DataLoader
    for i, (inputs, labels) in enumerate(original_data_loader):
        if i >= num_batches:  # Limit to desired number of batches
            break
        # Collect data
        if contaminate: 
            flip_mask = torch.rand(labels.size()) < 0.2 # 10% chance of flip label
            labels = torch.where(flip_mask, 1 - labels, labels)
        sampled_data.append((inputs, labels))

    # Flatten the list of tuples to create a new dataset
    new_inputs = torch.cat([inputs for inputs, _ in sampled_data])
    new_labels = torch.cat([labels for _, labels in sampled_data])

    # Create a new TensorDataset
    new_dataset = TensorDataset(new_inputs, new_labels)

    # Create a new DataLoader from the new dataset
    return new_dataset

def take_highest_influence(original_data_loader : DataLoader, num_datapoints : int, influences : List[float], seed) -> TensorDataset:
    torch.manual_seed(seed)
    original_dataset = original_data_loader.dataset
    sorted_indices = sorted(range(len(influences)), key=lambda i: influences[i], reverse=True)
    top_influence_indices = sorted_indices[:num_datapoints]

    top_influence_dataset = Subset(original_dataset, top_influence_indices)
    
    return top_influence_dataset

def smooth_distribution(probs, d, alpha=0.5):
        # Step 1: Sort indices of probs in descending order and select the top d
        sorted_indices = np.argsort(probs)[::-1]
        top_d_indices = sorted_indices[:d]
        
        # Step 2: Create a distribution retaining original probabilities for the top d indices
        top_d_distribution = np.zeros_like(probs)
        top_d_distribution[top_d_indices] = probs[top_d_indices]
        
        # Normalize the top_d_distribution to sum to 1 (only within top d elements)
        top_d_distribution /= top_d_distribution.sum()
        
        # Step 3: Create a fully uniform distribution
        fully_uniform = np.ones_like(probs) / len(probs)
        
        # Step 4: Interpolate between top-d distribution and fully uniform distribution based on alpha
        new_distribution = (1 - alpha) * top_d_distribution + alpha * fully_uniform
        
        return new_distribution



def sample_from_influence(original_data_loader : DataLoader, num_datapoints : int, influences : List[float], seed) -> TensorDataset:

    torch.manual_seed(seed)
    original_dataset = original_data_loader.dataset
    
    # randomly sample from datapoints based on influence
    normalized_influences = influences.tolist() # normalized
    normalized_influences = [x + abs(min(normalized_influences)) for x in normalized_influences] # normalized
    normalized_influences = [x/sum(normalized_influences) for x in normalized_influences] # normalized

    indices = np.random.choice(len(normalized_influences), size=num_datapoints, p=normalized_influences)

    top_influence_dataset = Subset(original_dataset, indices)
    
    return top_influence_dataset

def remove_lowest_influence_then_sample_uniformly(original_data_loader : DataLoader, num_datapoints : int, influences : torch.tensor, seed) -> TensorDataset:
    torch.manual_seed(seed)
    original_dataset = original_data_loader.dataset
    
    normalized_influences = deepcopy(influences)
    # remove lowest 10% data
    normalized_influences += abs(min(normalized_influences)) # make everything more than 0
    percentile_value = torch.quantile(influences, 0.1).item()
    # Set values below the 10th percentile to zero
    normalized_influences = normalized_influences.numpy()
    normalized_influences[normalized_influences < percentile_value] = 0
    
    num_harmful_points = sum(i == 0 for i in normalized_influences)
    
    # random sample from the rest of useful data uniformly
    normalized_influences[normalized_influences!=0] = 1/(len(normalized_influences) - num_harmful_points)
    indices = np.random.choice(len(normalized_influences), size=num_datapoints, p=normalized_influences) # sample from new list

    top_influence_dataset = Subset(original_dataset, indices)
    return top_influence_dataset

def remove_lowest_influence_then_sample_based_on_IF(original_data_loader : DataLoader, num_datapoints : int, influences : List[float], seed) -> TensorDataset:
    torch.manual_seed(seed)
    original_dataset = original_data_loader.dataset
    
    # remove lowest 10% data
    # Set values below the 10th percentile to zero
    normalized_influences = deepcopy(influences)
    normalized_influences += abs(min(normalized_influences)) # make everything more than 0
    percentile_value = torch.quantile(normalized_influences, 0.1).item()
    normalized_influences = normalized_influences.numpy()
    normalized_influences[normalized_influences < percentile_value] = 0
    
    
    # randomly sample from datapoints based on influence
    normalized_influences = normalized_influences.tolist() # normalized
    normalized_influences = [x/sum(normalized_influences) for x in normalized_influences] # normalized

    indices = np.random.choice(len(normalized_influences), size=num_datapoints, p=normalized_influences) # sample from new list

    top_influence_dataset = Subset(original_dataset, indices)
    return top_influence_dataset

def remove_tail_ends_then_uniform(original_data_loader : DataLoader, num_datapoints : int, influences : List[float], seed) -> TensorDataset:
    torch.manual_seed(seed)
    original_dataset = original_data_loader.dataset
    
    normalized_influences = deepcopy(influences)
    normalized_influences += abs(min(normalized_influences)) # make everything more than 0
    lower_percentile_value = torch.quantile(normalized_influences, 0.1).item()
    upper_percentile_value = torch.quantile(normalized_influences, 0.9).item()
    # Set values below and abovethe 10th percentile to zero
    normalized_influences = normalized_influences.numpy()
    normalized_influences[normalized_influences < lower_percentile_value] = 0
    normalized_influences[normalized_influences > upper_percentile_value] = 0
    
    # random sample from the rest of useful data uniformly
    normalized_influences[normalized_influences!=0] = 1 # set to 1, then normalize
    normalized_influences = [x/sum(normalized_influences) for x in normalized_influences] # normalized
    indices = np.random.choice(len(normalized_influences), size=num_datapoints, p=normalized_influences) # sample from new list

    top_influence_dataset = Subset(original_dataset, indices)
    return top_influence_dataset
        
# given a mixing ratio, sample from the data sources randomly to fulfill that amount
def sample_from(loaders : List[DataLoader], mixing_ratio, method="random_sample", seed=None, additional_info=None, base_number_of_batches=10, batch_size=16, shuffle=True, contaminate=False) -> DataLoader:
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        # Set the seed in PyTorch
        torch.manual_seed(seed)
    else:
        torch.manual_seed(seed)
    assert len(loaders) == len(mixing_ratio)
    sampled_data = []
    
    # "random_sample", # randomly sample
    # "highest_influence", # take top K influence
    # "lowest_influence", # take bottom K influence 
    # "influence_sample", # sample based on influence distribution
    # "reverse_influence_sample", # sample based on reverse influence distribution
    # "remove_harmful_then_uniform", # remove bottom 10% influence, then sample from the rest uniformly
    # "remove_harmful_then_follow_IF", # remove bottom 10% influence, then sample from the rest based on influence distribution
    # "remove_tail_ends_then_uniform",
    
    for idx, loader in enumerate(loaders):
        ratio = mixing_ratio[idx]
        if ratio == 0:
            continue
        if method == "random_sample":
            seed = random.randint(0,10000)
            sampled_data.append(sample_from_dataloader(loader, num_batches=int(ratio*base_number_of_batches), contaminate=contaminate, seed=seed))
        if method == "highest_influence":
            sampled_data.append(take_highest_influence(loader, num_datapoints=int(ratio*base_number_of_batches)*batch_size, influences=additional_info[idx], seed=seed))
        if method == "influence_sample":
            sampled_data.append(sample_from_influence(loader, num_datapoints=int(ratio*base_number_of_batches)*batch_size, influences=additional_info[idx], seed=seed))
        if method == "remove_harmful_then_uniform":
            sampled_data.append(remove_lowest_influence_then_sample_uniformly(loader, num_datapoints=int(ratio*base_number_of_batches)*batch_size, influences=additional_info[idx], seed=seed))
        if method == "remove_harmful_then_follow_IF":
            sampled_data.append(remove_lowest_influence_then_sample_based_on_IF(loader, num_datapoints=int(ratio*base_number_of_batches)*batch_size, influences=additional_info[idx], seed=seed))
        if method == "lowest_influence": # just multiply influence by -1
            sampled_data.append(take_highest_influence(loader, num_datapoints=int(ratio*base_number_of_batches)*batch_size, influences=-additional_info[idx], seed=seed))
        if method == "reverse_influence_sample":  # just multiply by -1
            sampled_data.append(sample_from_influence(loader, num_datapoints=int(ratio*base_number_of_batches)*batch_size, influences=-additional_info[idx], seed=seed))
        if method == "remove_tail_ends_then_uniform":
            sampled_data.append(remove_tail_ends_then_uniform(loader, num_datapoints=int(ratio*base_number_of_batches)*batch_size, influences=additional_info[idx], seed=seed))
        
    combined_dataset = ConcatDataset(sampled_data)
    combined_dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=shuffle) # keep same batch size
    
    return combined_dataloader

def flip_labels_in_dataset(dataset, num_classes, prob=0.1, seed=0):
    torch.seed(seed)
    """
    Randomly flips labels in a TensorDataset with a given probability.
    
    Args:
        dataset (TensorDataset): The original dataset containing features and labels.
        num_classes (int): The number of unique classes.
        prob (float): The probability of flipping a label.
    
    Returns:
        TensorDataset: A new TensorDataset with flipped labels.
    """
    # Extract features and labels from the dataset
    features, labels = dataset.tensors
    
    # Generate a mask for labels to be flipped (True means flip)
    flip_mask = torch.rand(labels.size()) < prob
    
    # Generate new random labels for the flipped ones
    new_labels = torch.randint(0, num_classes, labels.size())
    
    # Apply flipping based on the flip_mask
    flipped_labels = torch.where(flip_mask, new_labels, labels)
    
    # Return a new TensorDataset with the original features and flipped labels
    return TensorDataset(features, flipped_labels)