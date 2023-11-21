import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import numpy as np
from helper_functions import plot_decision_boundary,accuracy_fn

RANDOM_SEED=42
FEATURES_NUMBER=2
CLASSES_NUMBER=4
SAMPLES_NUMBER=1000

X,y=make_blobs(n_samples=SAMPLES_NUMBER,
               n_features=FEATURES_NUMBER,
               centers=CLASSES_NUMBER,
               random_state=RANDOM_SEED,
               cluster_std=1.5) # give the clusters a little shake up
#convert numpy to tensor
X=torch.from_numpy(X).type(torch.float)
y=torch.from_numpy(y).type(torch.LongTensor)
x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=RANDOM_SEED,test_size=0.2)
#visualize the dataset
plt.figure(figsize=(12,6))
plt.scatter(x=X[:,0],y=X[:,1],c=y,cmap=plt.cm.RdYlBu)
plt.show()
#check device
device="cuda" if torch.cuda.is_available() else "cpu"
class makeBlobsModel(nn.Module):
    def __init__(self,input_features,output_features,hidden_units=128):
        super().__init__()
        self.model=nn.Sequential(
        nn.Linear(in_features=input_features,out_features=hidden_units),
        nn.ReLU(),
        nn.Linear(in_features=hidden_units,out_features=hidden_units),
        nn.ReLU(),
        nn.Linear(in_features=hidden_units,out_features=hidden_units),
        nn.Softplus(),
        nn.Linear(in_features=hidden_units,out_features=output_features),
        )
    def forward(self,x:torch.Tensor)-> torch.Tensor:
        return self.model(x)
torch.manual_seed(RANDOM_SEED)    
model_0=makeBlobsModel(x_train.shape[1],len(torch.unique(y_train)))
# print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
#cpu to cuda
x_train,x_test=x_train,x_test
y_train,y_test=y_train,y_test
#create loss function and optimizer
learning_rate=0.01
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(lr=learning_rate,params=model_0.parameters())
model_0.eval()
with torch.inference_mode():
    test_logit=model_0(x_test)
    test_pred=torch.argmax(torch.softmax(test_logit,dim=1),dim=1)
# print(torch.softmax(test_logit,dim=1))
# print(test_pred[:5])
# plt.figure(figsize=(12,6))
# plot_decision_boundary(model_0,x_test,y_test)
# plt.show()  
# print(torch.sigmoid(test_logit)[:10])
# print(test_pred[:10])
# print(x_train[0],x_test[0],y_train[0],y_test[0])
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
epochs=1000
train_loss=[]
tst_loss=[]
epoch_val=[]
train_acc_val=[]
test_acc_val=[]
for epoch in range(epochs):
    model_0.train()
    epoch_val.append(epoch)
    y_logit=model_0(x_train)
    y_pred=torch.argmax(torch.sigmoid(y_logit),dim=1)
    
    #calculate the loss
    loss=loss_fn(y_logit,y_train)
    train_loss.append(loss.item())
    #calculate the accuracy
    acc=accuracy_fn(y_train,y_pred)
    train_acc_val.append(acc)
    #calculate zero grad
    optimizer.zero_grad()
    #do the backward
    loss.backward()
    
    #calculate the step
    optimizer.step()
    model_0.eval()
    with torch.inference_mode():
        test_logit=model_0(x_test)
        test_pred=torch.argmax(torch.sigmoid(test_logit),dim=1)
        
        test_loss=loss_fn(test_logit,y_test)
        tst_loss.append(test_loss.item())
        test_accuarcy=accuracy_fn(y_test,test_pred)
        test_acc_val.append(test_accuarcy)
        if epoch%10==0:
            print(f"epoch: {epoch}, train loss{loss:.5f},train acc{acc:.2f}%, test loss: {test_loss:.5f}, test acc : {test_accuarcy:.2f}%")

print(torch.softmax(test_logit,axis=1)[:20])
print(y_test[:20])

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("train")
plot_decision_boundary(model_0,x_train,y_train)
plt.subplot(1,2,2)
plt.title("test")
plot_decision_boundary(model_0,x_test,y_test)
plt.show()



# fig,ax=plt.subplots(nrows=2,ncols=2)

# ax[0,0].plot(epoch_val,train_loss)
# ax[0,0].set_title("train loss")
# ax[0,0].set_xlabel("epoch")
# ax[0,0].set_ylabel("loss")



# ax[0,1].plot(epoch_val,tst_loss)
# ax[0,1].set_title("test loss")
# ax[0,1].set_xlabel("epoch")
# ax[0,1].set_ylabel("loss")


# ax[1,0].plot(epoch_val,train_acc_val)
# ax[1,0].set_title("train acc")
# ax[1,0].set_xlabel("epoch")
# ax[1,0].set_ylabel("accuarcy")

# ax[1,1].plot(epoch_val,test_acc_val)
# ax[1,1].set_title("test acc")
# ax[1,1].set_xlabel("epoch")
# ax[1,1].set_ylabel("accuracy")


# plt.show()





