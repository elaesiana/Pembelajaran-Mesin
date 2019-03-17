
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


# In[2]:


#load dataset
data = pd.read_csv('data_iris.csv')
data = shuffle(data)


# In[3]:


#bagi dataset menjadi data training dan data validasi
X = np.array(data.iloc[:120,:4].T)
Y = np.array(data.iloc[:120, -3:-1].T)

Xval = np.array(data.iloc[120:, :4].T)
Yval = np.array(data.iloc[120:, -3:-1].T)

print("Shape of X : "+ str(X.shape))
print("Shape of Y : "+ str(Y.shape))
print("Shape of X_validation : "+ str(Xval.shape))
print("Shape of Y_validation : "+ str(Yval.shape))


# Neural network yang akan digunakan adalah neural network dengan 1 hidden layer

# In[4]:


n_x = X.shape[0] # jumlah node pada input layer
n_h = 4 #jumlah node pada hidden layer
n_y = Y.shape[0] # jumlah node pada output layer


# In[27]:


#inisialisasi parameter W1,b1,W2,b2
def inisialisasi(n_x, n_h, n_y):
    np.random.seed(5)
    W1 = np.random.rand(n_h, n_x) * 0.1
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y, n_h) * 0.1
    b2 = np.zeros((n_y,1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters


# In[7]:


def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s


# In[8]:


#fungsi untuk melakukan forward propagation
def forward(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    #menghitung fungsi aktivasi
    Z1 = np.dot(W1,X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)
    
    value = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, value


# In[9]:


#fungsi untuk menghitung cost/error function
def cost(A2, Y, parameters):
    m = Y.shape[1]
    
    cost_function = -1/(2*m)*np.sum(Y*np.log(A2) + ((1-Y)*np.log(1-A2)))
    
    return cost_function


# In[10]:


#fungsi untuk melakukan backward propagation
def backward(parameters, value, X, Y):
    m = X.shape[1]
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    A1 = value['A1']
    A2 = value['A2']
    
    #menghitung turunan dari dZ2,dW2,db2,dZ1,dW1,db1
    dZ2 = A2 - Y
    dW2 = 1/m*np.dot(dZ2,A1.T)
    db2 = 1/m*np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T,dZ2)*(A1*(1-A1)) #sigmoid function
    dW1 = 1/m*np.dot(dZ1,X.T)
    db1 = 1/m*np.sum(dZ1, axis=1, keepdims=True)
    
    deriv = {"dW1": dW1,
             "dW2": dW2,
             "db1": db1,
             "db2": db2}
    return deriv


# In[11]:


#fungsi untuk update parameter
def update(parameters, deriv, alpha):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    dW1 = deriv['dW1']
    db1 = deriv['db1']
    dW2 = deriv['dW2']
    db2 = deriv['db2']
    
    #update parameter
    W1 = W1 - alpha*dW1
    b1 = b1 - alpha*db1
    W2 = W2 - alpha*dW2
    b2 = b2 - alpha*db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters


# In[12]:


#fungsi untuk melakukan prediksi
def predict(parameters, X):
    A2, value = forward(X,parameters)
    
    #output berupa True jika A2>0.5, dan False jika A2<=0.5
    predict = (A2>0.5)
    #mengubah hasil prediksi menjadi 0,1,2
    predict = (predict[0].astype('int')*1) + (predict[1].astype('int')*2)
    #predict = predict.reshape((1,X.shape[1]))
    return predict


# In[13]:


#menghitung akurasi
def accuracy(Y, predictions):
    m = Y.shape[1]
    Y = Y[0]*1 + Y[1]*2
    acc = np.count_nonzero((predictions-Y)==0)/m
    return acc


# In[14]:


def model(X,Y, Xval, Yval,parameters,alpha):
    
    #np.random.seed(3)
    n_x = X.shape[0]
    n_y = Y.shape[0]
    
    #parameters = inisialisasi(n_x, n_h, n_y)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    #melakukan forward propagation
    A2, value = forward(X,parameters)
    val_A2, val_Value = forward(Xval, parameters)

    #melakukan backpropagation
    deriv = backward(parameters, value, X, Y)

    #melakukan prediksi
    train_predict = predict(parameters, X)
    val_predict = predict(parameters, Xval)

    #menghitung cost function
    train_cost = cost(A2, Y, parameters)
    val_cost = cost(val_A2,Yval, parameters)

    #menghitung akurasi
    train_akurasi = accuracy(Y, train_predict)
    val_akurasi = accuracy(Yval, val_predict)

    #update parameter
    parameters = update(parameters, deriv, alpha)
    
    results = {"parameters": parameters,
              "train_predict": train_predict,
              "val_predict": val_predict,
              "train_error": train_cost,
              "val_error": val_cost,
              "train_akurasi": train_akurasi,
              "val_akurasi": val_akurasi,
              "W1": parameters['W1'],
              "b1": parameters['b1'],
              "W2": parameters['W2'],
              "b2": parameters['b2']}
          
    return results


# In[63]:


train_error = []
validation_error = []
train_accuracy = []
validation_accuracy = []

# untuk n_h = 4
n_h = 4
parameters = inisialisasi(n_x, n_h, n_y)
alpha = 0.1

#n_epoch = 5000
for i in range(5000):
    results = model(X,Y,Xval, Yval, parameters, alpha)
    parameters = results['parameters']
    train_error.append(results['train_error'])
    validation_error.append(results['val_error'])
    train_accuracy.append(results['train_akurasi'])
    validation_accuracy.append(results['val_akurasi'])


# In[54]:


#menampilkan hasil prediksi dan data asli
print(results['train_predict'])

Y2 = Y[0]*1 + Y[1]*2
print('\n',Y2)


# In[40]:


# Visualisasi Error Function
plt.plot(train_error)
plt.plot(validation_error)
plt.xlabel('Iterasi')
plt.ylabel('Error')
plt.legend(labels=['training','validation'])
plt.title("Kurva Error")


# In[41]:


# Visualisasi Akurasi
plt.plot(train_accuracy)
plt.plot(validation_accuracy)
plt.xlabel('Iterasi')
plt.ylabel('Accuracy')
plt.legend(labels=['training','validation'])
plt.title('Kurva Akurasi')


# In[75]:


train_error = []
validation_error = []
train_accuracy = []
validation_accuracy = []

# untuk n_h = 4
parameters = inisialisasi(n_x, n_h, n_y)
alpha = 0.8
#n_epoch = 5000
for i in range(5000):
    results = model(X,Y,Xval, Yval, parameters, alpha)
    parameters = results['parameters']
    train_error.append(results['train_error'])
    validation_error.append(results['val_error'])
    train_accuracy.append(results['train_akurasi'])
    validation_accuracy.append(results['val_akurasi'])


# In[76]:


# Visualisasi Error Function
plt.plot(train_error)
plt.plot(validation_error)
plt.xlabel('Iterasi')
plt.ylabel('Error')
plt.legend(labels=['training','validation'])
plt.title('Kurva Error')


# In[77]:


# Visualisasi Akurasi
plt.plot(train_accuracy)
plt.plot(validation_accuracy)
plt.xlabel('Iterasi')
plt.ylabel('Accuracy')
plt.legend(labels=['training','validation'])
plt.title('Kurva Akurasi')

