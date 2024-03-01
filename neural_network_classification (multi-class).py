import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy

# Set the random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Define dataset parameters
NUM_SAMPLES_PER_CLASS = 100
NUM_FEATURES = 2
NUM_CLASSES = 3

# Initialize the dataset
X = np.zeros((NUM_SAMPLES_PER_CLASS * NUM_CLASSES, NUM_FEATURES))
y = np.zeros(NUM_SAMPLES_PER_CLASS * NUM_CLASSES, dtype='uint8')

# Generate spiral data
for class_index in range(NUM_CLASSES):
    indices = range(NUM_SAMPLES_PER_CLASS * class_index, NUM_SAMPLES_PER_CLASS * (class_index + 1))
    radius = np.linspace(0.0, 1, NUM_SAMPLES_PER_CLASS)  # Radius
    theta = np.linspace(class_index * 4, (class_index + 1) * 4, NUM_SAMPLES_PER_CLASS) + np.random.randn(NUM_SAMPLES_PER_CLASS) * 0.2  # Angle
    X[indices] = np.c_[radius * np.sin(theta), radius * np.cos(theta)]
    y[indices] = class_index

# Visualize the dataset
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
plt.show()

# Convert the dataset to PyTorch tensors
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long()

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

# Move data to the appropriate device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)

# Define the model
class SpiralModel(nn.Module):
    def __init__(self):
        super(SpiralModel, self).__init__()
        self.layer1 = nn.Linear(NUM_FEATURES, 10)
        self.layer2 = nn.Linear(10, 10)
        self.output_layer = nn.Linear(10, NUM_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output_layer(x)
        return x

model = SpiralModel().to(device)

# Define loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

# Define accuracy metric
accuracy_metric = Accuracy(num_classes=NUM_CLASSES).to(device)

# Training loop
EPOCHS = 1000
for epoch in range(EPOCHS):
    # Forward pass
    predictions = model(X_train)
    loss = loss_function(predictions, y_train)
    train_accuracy = accuracy_metric(predictions.argmax(dim=1), y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Evaluation on the test set
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test)
        test_loss = loss_function(test_predictions, y_test)
        test_accuracy = accuracy_metric(test_predictions.argmax(dim=1), y_test)
    
    model.train()

    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.2f}, Accuracy: {train_accuracy:.2f} | Test Loss: {test_loss:.2f}, Test Accuracy: {test_accuracy:.2f}")

def plot_decision_boundary(model, X, y):
  
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Refer to: https://madewithml.com/courses/foundations/neural-networks/ 
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), 
                         np.linspace(y_min, y_max, 101))

    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # predict
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1) # mutli-class
    else: 
        y_pred = torch.round(torch.sigmoid(y_logits)) # binary
    
    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_1, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_1, X_test, y_test)
