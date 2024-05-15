#%%
from sklearn.datasets import fetch_california_housing
import pandas as pd

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['price'] = data.target
#%%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)
#%%
import torch
import torch.nn as nn
import torch.optim as optim

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.lin(x)
        return out

input_dim = X_train.shape[1]
output_dim = 1

model = LinearRegression(input_dim, output_dim)
#%%
criterion = nn.MSELoss()
learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
#%%
num_epochs = 1000

for epoch in range(num_epochs):
    # Convert numpy arrays to PyTorch tensors
    inputs = torch.from_numpy(X_train).float()
    labels = torch.from_numpy(y_train).float()

    # Clear the gradients
    optimizer.zero_grad()

    # Forward pass, compute the loss
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # Backward pass and update the weights
    loss.backward()
    optimizer.step()

    print('Epoch [{:d}/{:d}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
#%%
# Convert numpy arrays to PyTorch tensors
inputs = torch.from_numpy(X_test).float()
labels = torch.from_numpy(y_test).float()

# Deactivate the dropout
model.eval()

# Forward pass
outputs = model(inputs)

# Compute the loss
loss = criterion(outputs, labels)

print('Test loss: {:.4f}'.format(loss.item()))
#%%
import matplotlib.pyplot as plt
import numpy as np

# Make predictions on the test set
with torch.no_grad():
    test_outputs = model(torch.from_numpy(X_test).float())

# Convert PyTorch tensors to numpy arrays
predicted_values = test_outputs.numpy()
true_values = y_test

# Create a scatter plot for true values vs predicted values
plt.scatter(true_values, predicted_values)

# Add a line representing perfect predictions (y = x)
perfect_line = np.linspace(min(true_values), max(true_values), 100)
plt.plot(perfect_line, perfect_line, color='red')

# Add labels and title
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True Values vs Predicted Values')

# Show the plot
plt.show()

# %%
