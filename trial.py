import pandas as pd
import torch

# Load the dataset
df = pd.read_csv("insurance.csv")

# Preprocess the data
t_sex = torch.tensor(pd.get_dummies(df["sex"]).values, dtype=torch.float32)
t_smoker = torch.tensor(pd.get_dummies(df["smoker"]).values, dtype=torch.float32)
t_region = torch.tensor(pd.get_dummies(df["region"]).values, dtype=torch.float32)

t_age = torch.tensor(df["age"].values, dtype=torch.float32).unsqueeze(1)/100
t_bmi = torch.tensor(df["bmi"].values, dtype=torch.float32).unsqueeze(1)/100
t_children = torch.tensor(df["children"].values, dtype=torch.float32).unsqueeze(1)/10
t_charges = torch.tensor(df["charges"].values, dtype=torch.float32).unsqueeze(1)/50000

t_input = torch.cat([t_age, t_bmi, t_children, t_sex, t_smoker, t_region], dim=1)
# Define your neural network model sigmoid if answer between 1 and 0
def build_model():
    return torch.nn.Sequential(
        torch.nn.Linear(11, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 1)
    )

from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in kf.split(t_input):
    t_train_input = t_input[train_index]
    t_train_target = t_charges[train_index]
    t_test_input = t_input[test_index]
    t_test_target = t_charges[test_index]

    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(100):
        optimizer.zero_grad()
        t_train_output = model(t_train_input)
        loss = loss_fn(t_train_output, t_train_target)
        loss.backward()
        optimizer.step()
        print(epoch, loss)

    t_test_output = model(t_test_input)
    test_loss = loss_fn(t_test_output, t_test_target)
    print(f"Test loss: {test_loss}")

# separate 80/20 train and test
from sklearn.model_selection import train_test_split
t_train_input, t_test_input, t_train_target, t_test_target = train_test_split(t_input, t_charges, test_size=0.2, random_state=42)

model = build_model()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

losses = []
test_losses = []
for epoch in range(100):
    optimizer.zero_grad()
    t_train_output = model(t_train_input)
    loss = loss_fn(t_train_output, t_train_target)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    t_test_output = model(t_test_input)
    test_loss = loss_fn(t_test_output, t_test_target)
    test_losses.append(test_loss.item())

import matplotlib.pyplot as plt
plt.plot(losses, label="train")
plt.plot(test_losses, label="test")
plt.legend()
plt.show()

t_pred = model(t_test_input).detach() * 50000
t_test_target = t_test_target * 50000

df_pred = pd.DataFrame(torch.cat([t_test_target, t_pred], dim=1).numpy(), columns=["actual", "prediction"])
print(df_pred)

df_pred['diff'] = df_pred['prediction'] - df_pred['actual']
print(df_pred)

#compute mean squared error
print(df_pred['diff'].pow(2).mean())
#root mean squared error
print(df_pred['diff'].pow(2).mean().sqrt())

#plot prediction vs actual
plt.scatter(t_test_target, t_pred)
plt.xlabel("Actual")
plt.ylabel("Prediction")
plt.show()

plt.plot(t_test_target, label="actual")
plt.plot(t_pred, label="prediction")
plt.legend()
plt.show()



