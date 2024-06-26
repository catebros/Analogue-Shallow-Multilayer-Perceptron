import torch
import pandas as pd
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class NeuralNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_neurons):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(num_inputs, num_neurons)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(num_neurons, num_outputs)
        self.sigmoid = nn.Sigmoid()  

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        x = self.sigmoid(x) 
        return x

def train_model(dataset_path, num_neurons):
    data = pd.read_csv(dataset_path, header=None) 
    X = data.iloc[:, :-1].values  
    y = data.iloc[:, -1].values   

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Create the model
    model = NeuralNetwork(X_train.shape[1], 1, num_neurons)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        output = torch.clamp(output, 1e-7, 1 - 1e-7)  
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            predictions = model(X_test)
            predictions = (predictions > 0.5).float()
            accuracy = (predictions.eq(y_test).sum() / float(y_test.shape[0])).item()

        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}')

    return accuracy

if __name__ == '__main__':
    accuracies = {}
    for num_neurons in [2, 3, 4]:
        print(f"Training model with {num_neurons} middle layer neurons")
        accuracy = train_model('train.csv', num_neurons)
        accuracies[num_neurons] = accuracy

    print("Accuracies for different configurations:", accuracies)

