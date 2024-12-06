import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ANN model with one hidden layer
class ANNModel(nn.Module):
    def __init__(self, input, neuron, output):
        super(ANNModel, self).__init__()
        # input layer to hidden layer
        self.fc1 = nn.Linear(input, neuron)
        # hidden to output layer
        self.fc2 = nn.Linear(neuron, output)
        # activation func
        self.relu = nn.ReLU()

    def forward(self, x):
        # activation func for hidden layer
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# read dataset
data = pd.read_csv("maternal_health_risk_data_set.csv")


features = data[
    ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
].values
# get and label target as integer values
target = data["RiskLevel"].map({"low risk": 0, "mid risk": 1, "high risk": 2}).values


# min-max scaling normalization
features_min = features.min(axis=0)
features_max = features.max(axis=0)
# feature values will range from 0 to 1
features = (features - features_min) / (features_max - features_min)


# convert to pytorch tensors
feature_tensors = torch.FloatTensor(features)
target_tensors = torch.LongTensor(target)


# K-fold cross-validation
K = 10
fold_size = len(feature_tensors) // K
fold_results = []
epochs = 500

# shuffle index
indices = np.arange(len(feature_tensors))
np.random.shuffle(indices)

for fold in range(K):
    # prepare train and test indices
    test_indices = indices[fold * fold_size : (fold + 1) * fold_size]
    train_indices = np.concatenate(
        (indices[: fold * fold_size], indices[(fold + 1) * fold_size :])
    )

    # training and testing datasets
    train_dataset = feature_tensors[train_indices]
    train_target_dataset = target_tensors[train_indices]
    test_dataset = feature_tensors[test_indices]
    test_target_dataset = target_tensors[test_indices]

    model = ANNModel(6, 32, 3)
    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # learning rate 0.1
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    # train the model
    for epoch in range(epochs):
        model.train()
        # clear gradients before backpropagation
        optimizer.zero_grad()
        # forward propagation
        outputs = model(train_dataset)

        loss = criterion(outputs, train_target_dataset)
        loss.backward()
        # update weights and biases
        optimizer.step()

    # evaluate on test dataset
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_dataset)
        _, predicted_classes = torch.max(test_outputs.data, 1)
        siz = len(test_target_dataset)
        accuracy = (test_target_dataset == predicted_classes).sum().item() / siz

        print(f"Accuracy result on Fold {fold + 1}: {accuracy:.2f}")
        fold_results.append(accuracy)


average_accuracy = np.mean(fold_results)
print(
    f"\nAverage accuracy rate across {K} folds with 6 inputs, 32 neurons in hidden "
    + f"layer, and 3 outputs in {epochs} epochs: {average_accuracy:.2f}\n"
)
