import numpy as np
import random
import os
from helper_functions import plot_decision_boundary,accuracy_fn,plot_predictions
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
Dir="csv"
f_n="spiral.csv"
path=os.path.join(Dir,f_n)
df=pd.read_csv(path)
df=df.sample(frac=1).reset_index(drop=True)
print(df.head())
# Data shapes
X = df[["X", "Y"]].values
y = df["Group"].values
print ("X: ", np.shape(X))
print ("y: ", np.shape(y))
plt.figure(figsize=(6,12))
plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.RdYlBu)
plt.show()
device="cuda" if torch.cuda.is_available() else "cpu"
X=torch.from_numpy(X).type(torch.float)
y=torch.from_numpy(y).type(torch.float)
x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)
print(x_train.shape[1])
class SpiralModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(in_features=2,out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=3)
        )
    def forward(self,x:torch.Tensor)-> torch.Tensor:
        return self.model(x)


epochs=1000
torch.cuda.manual_seed(42)
model_0=SpiralModel().to(device)
x_train,x_test=x_train.to(device),x_test.to(device)
y_train,y_test=y_train.to(device),y_test.to(device)


loss_fn=nn.BCEWithLogitsLoss()
learning_rate=0.01
optimizer=torch.optim.Adam(lr=learning_rate,params=model_0.parameters())

for epoch in range(epochs):
    model_0.train()
    y_logit=model_0(x_train).squeeze()
    y_pred=torch.round(torch.sigmoid(y_logit))
    
    loss=loss_fn(y_logit,y_train)
    acc=accuracy_fn(y_pred=y_pred,y_true=y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
model_0.eval()
with torch.inference_mode():
    test_logit=model_0(x_test).squeeze()
    test_pred=torch.round(torch.sigmoid(test_logit))
    
    loss_test=loss_fn(test_logit,y_test)
    acc_test=accuracy_fn(test_pred,y_test)
print(f"test loss: {loss:.5f}, train accuarcy: {acc:.2f}% , test loss: {loss_test:.5f}, test accuarcy {acc_test:.2f}%")
    

