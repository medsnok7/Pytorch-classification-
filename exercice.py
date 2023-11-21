from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from helper_functions import plot_decision_boundary,accuracy_fn
RANDOM_SEED=42
N_SAMPLES=1000

X,y=make_moons(n_samples=N_SAMPLES,
               random_state=RANDOM_SEED,
               noise=0.04)
#convert numpy to tensor
X=torch.from_numpy(X).type(torch.float)
y=torch.from_numpy(y).type(torch.LongTensor)
#split the data
x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=RANDOM_SEED,test_size=0.2)
#visualisation 
# plt.figure(figsize=(12,6))
# plt.scatter(x=X[:,0],y=X[:,1],c=y,cmap=plt.cm.RdYlBu)
# plt.show()

# print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
device="cuda" if torch.cuda.is_available() else "cpu"
print(device)
#cpu to cuda
x_train,x_test=x_train.to(device),x_test.to(device)
y_train,y_test=y_train.to(device),y_test.to(device)

#create the class model
class makeMoonsModel(nn.Module):
    def __init__(self,input_features,output_features,hidden_layer1=32,hidden_layer2=64):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(in_features=input_features,out_features=hidden_layer1),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layer1,out_features=hidden_layer2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layer2,out_features=output_features)
        )
    def forward(self,x:torch.Tensor)-> torch.Tensor:
        return self.model(x)

in_feat=x_train.shape[1]
out_feat=1
print(in_feat,out_feat)
#model instance
torch.cuda.manual_seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
model_0=makeMoonsModel(input_features=in_feat,output_features=out_feat).to(device)

#evalute the parameters
# model_0.train()
# with torch.inference_mode():
#     test_logits=model_0(x_test)
#     test_pred=torch.round(torch.sigmoid(test_logits))

print(x_test.shape,y_test.shape)
plt.figure(figsize=(12,6))
plot_decision_boundary(model_0,x_test,y_test)
plt.show()
    
#train and test
epochs=1000
l_r=0.01
#set the loss and optimizer
loss_fn=nn.BCEWithLogitsLoss()
optimizer=torch.optim.Adam(lr=l_r,params=model_0.parameters())
#train the model
# for epoch in range(epochs):
    # y_logits=model_0(x_train)





