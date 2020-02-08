
# coding: utf-8

# In[117]:


import torch
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


# In[118]:


transform = transforms.Compose([
    transforms.ToTensor()
])


# In[119]:


import torchvision as thv
train = thv.datasets.MNIST('./', download=True, train=True,transform=transform)
val = thv.datasets.MNIST('./', download=True, train=False,transform=transform)
print(train.data.shape, len(train.targets))


# In[120]:


def get_indices(dataset,class_name_1,class_name_2):
    indices =  []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] == class_name_1 or dataset.targets[i] == class_name_2:
            indices.append(i)
    return indices
train_idx = get_indices(train, 0, 1)
val_idx = get_indices(val, 0, 1)
np.random.shuffle(train_idx)
np.random.shuffle(val_idx)
train_data_arr = np.asarray(train.data)[train_idx]
train_target_arr = np.asarray(train.targets)[train_idx]
val_data_arr = np.asarray(val.data)[val_idx]
val_target_arr = np.asarray(val.targets)[val_idx]


# In[121]:


print(type(train_data_arr))
print(type(train_target_arr))
print(type(val_data_arr))
print(type(val_target_arr))
print(train_data_arr.shape)
print(train_target_arr.shape)
print(val_data_arr.shape)
print(val_target_arr.shape)


# In[122]:


plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(train_data_arr[0:5], train_target_arr[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize = 20)


# In[123]:


print(train_data_arr.shape)


# In[124]:


def subsample(x):
    return x.reshape(-1, 28, 28)[:, ::2, ::2].reshape(-1, 14*14)


# In[125]:


train_data_arr = subsample(train_data_arr)
val_data_arr = subsample(val_data_arr)


# In[126]:


train_data_arr = train_data_arr/255
val_data_arr = val_data_arr/255


# In[127]:


y_i = np.ones([train_target_arr.shape[0]])
one = np.where(train_target_arr==1)
y_i[one] = -1
y_i = y_i.reshape(1,train_target_arr.shape[0])


# In[128]:


def initialize_para(dim):
    w = np.random.normal(loc=0, scale=1, size=(dim, 1))
    b = np.random.normal(1)
    return w, b


# In[129]:


def logistic_regression(x_i,y_i,w,w_0):
    n = x_i.shape[0]
    tmp_1 = (-y_i) * (np.matmul(w.T,x_i.T) + w_0)
    tmp_2 = np.log(1 + np.exp(tmp_1))
    l2 = (lamda/2)*(np.sum(w**2) + w_0**2)
    loss = (1/n)*np.sum(tmp_2) + l2
    return loss


# In[130]:


def backward(x_i,y_i,w,w_0):
    n = x_i.shape[0]
    tmp_1 = 1 + np.exp(y_i*(np.matmul(w.T,x_i.T) + w_0))
    dw = (1/n)*np.sum((x_i / tmp_1.T[:])*(-y_i.T[:]),axis=0)
    dw_0 = (1/n)*np.sum(-(y_i / tmp_1))
    return dw,dw_0


# In[131]:


total_loss = []
v_w = np.zeros([196, 1])
v_w_0 = 0
m = 0.8
eta = 0.1
lamda = 0.1
w,w_0 = initialize_para(196)


# In[132]:


for i in range(100):
    loss = logistic_regression(train_data_arr,y_i,w,w_0)
    total_loss.append(loss)
    dev_w = w-m*eta*v_w
    dev_w_0 = w_0-m*eta*v_w_0
    dw,dw_0 = backward(train_data_arr,y_i,dev_w,dev_w_0)
    dw = dw.reshape(196,1)
    dw = dw + lamda*w
    v_w = m*v_w + dw
    w = w - eta*v_w
    
    dw_0 = dw_0 + lamda*w_0
    v_w_0 = m*v_w_0 + dw_0
    w_0 = w_0 - eta*v_w_0


# In[133]:


x = np.arange(0,100,1)
fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.set_yscale('log')
plt.plot(x,y)

