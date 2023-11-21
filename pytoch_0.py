import torch
from torch import nn
import numpy
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import pandas as pd
import requests
import os
from pathlib import Path
from helper_functions import plot_decision_boundary,plot_predictions

from sklearn.model_selection import train_test_split
#make 1000 samples
samples=1000

#create circles
X,y=make_circles(samples,noise=0.03,random_state=42)
# print(X[:5],y[:5])

circles=pd.DataFrame({"X1":X[:,0],"X2":X[:,1],"label":y})
# print(circles.head(10))
# plt.scatter(x=X[:,0],y=X[:,1],c=y,cmap=plt.cm.RdYlBu)
# plt.show()

#turn data into tensors and create train and test splits
X=torch.from_numpy(X).type(torch.float)
y=torch.from_numpy(y).type(torch.float)
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# plot the training data
# plt.figure(figsize=(12,6))
# plt.scatter(x=x_train[:,0],y=x_train[:,1],c=y_train,cmap=plt.cm.RdYlBu)
# plt.show()

# print(len(x_train),len(x_test))
device="cuda" if torch.cuda.is_available() else "cpu"
# print(device)
def accuracy_fn(y_true,y_pred):
    correct=(torch.eq(y_true,y_pred).sum()).item()
    return ((correct/len(y_true))*100)
#build a model
class makeCircleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(in_features=2,out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=1)
        )
        
    def forward(self,x:torch.Tensor)-> torch.Tensor:
        return self.model(x)
torch.cuda.manual_seed(42)
model_0=makeCircleModel()
with torch.inference_mode():
    y_pred=model_0(x_test)
# print(f"predicted values {y_pred[:5]} \n labels : {y_test[:5]}")
#set loss and optimizer
plt.figure(figsize=(12,6))
plot_decision_boundary(model_0,x_train,y_train)
plt.show()
learning_rate=0.01
loss_fn=nn.BCEWithLogitsLoss()
optimizer=torch.optim.Adam(lr=learning_rate,params=model_0.parameters())
#convert to device
x_train,x_test=x_train,x_test
y_train,y_test=y_train,y_test
epochs=1000
epoch_values=[]
loss_values=[]
acc_values=[]
for epoch in range(epochs):
    model_0.train()
    epoch_values.append(epoch)
    #forward pass
    y_logit=model_0(x_train).squeeze()
    y_pred=torch.round(torch.sigmoid(y_logit))
    #calculate loss and accuracy
    loss=loss_fn(y_logit,y_train) #nn.BCEWithLogitsLoss expects raw logits aas input
    loss_values.append(loss.item())
    # loss=loss_fn(torch.sigmoid(y_logit).squeeze(), y_train.squeeze()) # nn.BCELoss expects prediciton probabilities as input
    acc=accuracy_fn(y_true=y_train,y_pred=y_pred)
    acc_values.append(acc)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_0.eval()
    with torch.inference_mode():
        test_logit=model_0(x_test).squeeze()
        test_pred=torch.round(torch.sigmoid(test_logit))

        test_loss=loss_fn(test_logit,y_test)
        test_acc=accuracy_fn(y_true=y_test,y_pred=test_pred)
#     if epoch%10==0:   
#         print(f"epoch : {epoch}, loss : {loss:.5f}, acc : {acc:.2f}%, test loss: {test_loss:.5f}, test accuracy: {test_acc:.2f}%")


# print(type(loss_values[0]),type(acc_values[0]))
model_0.eval()
with torch.inference_mode():
    test_logit=model_0(x_test).squeeze()
    test_pred=torch.round(torch.sigmoid(test_logit))
    loss=loss_fn(test_logit,y_test)
    acc=accuracy_fn(y_true=y_test,y_pred=test_pred)
plt.plot(epoch_values,loss_values,label="Loss ")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.show()
# #download helper functions from learn pytorch repo(if it's not already downloaded)

# if Path("helper_functions.py").is_file():
#     print("file already existes")
# else:
#     print("download helper_function.py")
#     request=requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
#     with open("helper_functions.py","wb") as f:
#         f.write(request.content)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("train")
plot_decision_boundary(model_0,x_train,y_train)
plt.subplot(1,2,2)
plt.title("test")
plot_decision_boundary(model_0,x_test,y_test)
plt.show()

#saving the model_0
pth=Path("models")
print("directory already exists")
pth.mkdir(exist_ok=True,parents=True)
file_name="model_0.pth"
path=os.path.join(pth,file_name)
torch.save(f=path,obj=model_0.state_dict())
#loading the model_0
# model_1=makeCircleModel()
# model_1.load_state_dict(torch.load(f=path))

# model_1.eval()
# with torch.inference_mode():
#     new_pred_test=torch.round(torch.sigmoid(model_1(x_test).squeeze()))
# plt.figure(figsize=(12,6))
# plt.title("train")
# plot_decision_boundary(model_1,x_test,
#                        new_pred_test)
# plt.show()

