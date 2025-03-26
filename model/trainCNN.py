import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn.functional as F

# Define the MultiViewCNN class
class MultiViewCNN(nn.Module):
    def __init__(self):
        super(MultiViewCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Example usage
if __name__ == "__main__":
    class RandomDataset(Dataset):
        def __init__(self, size, length):
            self.len = length
            self.data = torch.randn(length, 3, 32, 32)

        def __getitem__(self, index):
            return self.data[index], torch.tensor(0)

        def __len__(self):
            return self.len

    dataset = RandomDataset(1000, 1000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = MultiViewCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        for i, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/5], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    # Save the trained model's state dictionary
    torch.save(model.state_dict(), 'results/model.pth')