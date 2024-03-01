import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def load_data():
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def visualize_samples(dataset, n_samples=5):
    plt.figure(figsize=(10, 2))
    for i in range(n_samples):
        ax = plt.subplot(1, n_samples, i + 1)
        idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[idx]
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(label)
        ax.axis('off')
    plt.show()

class TinyVGG(nn.Module):
    def __init__(self):
        super(TinyVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(7*7*64, 256), nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 7*7*64)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, test_loader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        start_time = time.time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                test_loss += criterion(output, labels).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print(f'Epoch: {epoch+1}, Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)')
        print(f"Training time for epoch: {time.time() - start_time:.2f} seconds")

        model.train()

def visualize_predictions(model, dataloader, num_images=5):
    model.eval()  # Set the model to evaluation mode
    images, labels = next(iter(dataloader))  # Get a batch of images and labels
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    # Plot the images and predictions
    plt.figure(figsize=(10, 2))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i].cpu().numpy().squeeze(), cmap='gray')
        plt.title(f"Pred: {predicted[i]}, True: {labels[i]}")
        plt.axis('off')
    plt.show()
  
def plot_confusion_matrix(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()    

if __name__ == "__main__":
    train_dataset, test_dataset = load_data()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = TinyVGG().to(device)
    visualize_samples(train_dataset)
    train_model(model, train_loader, test_loader, epochs=5)
    visualize_predictions(model, test_loader)
    plot_confusion_matrix(model, test_loader)
