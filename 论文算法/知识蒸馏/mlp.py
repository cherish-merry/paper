import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.nn.functional as F



class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.scaler = StandardScaler()

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(input_size, 128)

        self.fc2 = nn.Linear(128, 256)

        self.fc3 = nn.Linear(256, 512)

        self.fc4 = nn.Linear(512, output_size)  # 输出维度为10

    def forward(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.relu(x)

        x = self.fc2(x)
        # x = self.bn2(x)
        x = self.relu(x)

        x = self.fc3(x)
        # x = self.bn3(x)
        x = self.relu(x)

        x = self.fc4(x)
        return x

    def fit(self, train_X, train_Y):
        train_X = self.scaler.fit_transform(train_X)
        # 修改输入通道数
        X_train_tensor = torch.FloatTensor(train_X)
        y_train_tensor = torch.LongTensor(train_Y)
        train_data = TensorDataset(X_train_tensor, y_train_tensor)
        batch_size = 64
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        learning_rate = 0.001
        num_epochs = 150
        # MLP 模型
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            self.train()
            for i, (inputs, labels) in enumerate(train_loader):
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # if (i + 1) % 10 == 0:
                #     print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}')

    def predict_proba(self, test_X, T):
        test_X = self.scaler.fit_transform(test_X)
        X_val_tensor = torch.FloatTensor(test_X)
        val_data = TensorDataset(X_val_tensor)
        val_loader = DataLoader(dataset=val_data, batch_size=64, shuffle=False)
        self.eval()
        all_probabilities = []
        for inputs in val_loader:
            output = self(inputs[0])  # inputs[0] 是数据，inputs[1] 是标签，但这里不需要标签
            softmax = F.softmax(output / T, dim=1)
            all_probabilities.append(softmax.detach().numpy())
        all_probabilities = np.concatenate(all_probabilities, axis=0)
        return all_probabilities

    def predict(self, test_X):
        test_X = self.scaler.transform(test_X)  # 这里不需要再调用fit_transform，只需要transform即可
        X_val_tensor = torch.FloatTensor(test_X)
        val_data = TensorDataset(X_val_tensor)
        val_loader = DataLoader(dataset=val_data, batch_size=64, shuffle=False)
        self.eval()
        all_predictions = []
        for inputs in val_loader:
            outputs = self(inputs[0])  # inputs[0] 是数据，inputs[1] 是标签，但这里不需要标签
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.append(predicted.detach().numpy())
        all_predictions = np.concatenate(all_predictions, axis=0)
        return all_predictions
