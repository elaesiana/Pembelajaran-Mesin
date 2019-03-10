import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# load dataset
data = pd.read_csv('data_iris.csv')
data = data[:100] #data yang diambil hanya 100 data, yaitu setosa dan versicolor
data = shuffle(data) #karena data dalam kondisi terurut, maka perlu di shuffle terlebih dahulu

def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s
    
# inisialisasi w dan b
def initialize(n): # n = jumlah feature
    w = np.zeros((n,1))
    b = 0
    return w,b
    
# fungsi untuk menghitung error function
def error_func(w,b,X,Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)
    #error = 1/m*(np.sum((A-Y)**2))
    error = -1/m*(np.sum(Y*np.log(A)+((1-Y)*np.log(1-A))))
    
    return error
    
# fungsi untuk menghitung dJ/dw dan dJ/db
def derivative(w,b,X,Y):
    A = sigmoid(np.dot(w.T,X)+b)
    m = X.shape[1]
    
    dw = 1/m*np.dot(X,(A-Y).T)
    db = 1/m*np.sum(A-Y)
    
    deriv = {"dw": dw,
             "db": db}
    return deriv

# fungsi untuk mengupdate parameter w dan b
def update(w,b,X,Y,alpha):
    errors = error_func(w,b,X,Y)
    deriv = derivative(w,b,X,Y)
        
    dw = deriv["dw"]
    db = deriv["db"]
        
    # update w and b
    w = w - (alpha*dw)
    b = b - (alpha*db)
            
    params = {"w": w,
              "b": b}
    deriv = {"dw": dw,
             "db": db}
    return params, deriv, errors

def predict(w,b,X):
    m = X.shape[1]
    prediction = np.zeros((1,m),dtype=int)
    w = w.reshape(X.shape[0],1)
    
    # menghitung probabilitas
    A = sigmoid(np.dot(w.T,X)+b)
    
    for i in range(A.shape[1]):
        if A[0,i] > 0.5:
            prediction[0,i] = int(1)
        else:
            prediction[0,i] = int(0)
        pass
    return prediction
    
def accuracy(prediction,Y):
    m = Y.shape[1]
    acc = np.count_nonzero((prediction-Y)==0)/m
    return acc
    
def model(data,w,b,alpha, K=5):
    size = int(len(data)/K)
    train_error = []
    validation_error = []
    train_accuracy = []
    validation_accuracy = []
    train_prediction = []
    validation_prediction = []
    
    for i in range(1,(K+1)):
        validation = data.iloc[(i-1)*size:i*size,:]
        train = data.iloc[i*size:, :]
        train = train.append(data.iloc[:(i-1)*size,:])
        
        train_x = np.array(train.iloc[:,:4]).T
        train_y = np.array(train.iloc[:, -1])
        train_y = train_y.reshape((1, train_y.shape[0]))
        
        validation_x = np.array(validation.iloc[:, :4]).T
        validation_y = np.array(validation.iloc[:, -1])
        validation_y = validation_y.reshape((1, validation_y.shape[0]))
           
        # prediksi pada data training dan validation
        train_predict = (predict(w,b,train_x))
        validation_predict = (predict(w,b,validation_x))
        
        train_prediction.append(train_predict)
        validation_prediction.append(validation_predict)
        
        # menghitung error function pada data training dan validation
        train_error.append(error_func(w,b,train_x, train_y))
        validation_error.append(error_func(w,b,validation_x, validation_y))
        
        #menghitung akurasi pada data training dan validation
        train_accuracy.append(accuracy(train_predict, train_y))
        validation_accuracy.append(accuracy(validation_predict, validation_y))
    
    # melakukan update parameter w dan b
    params, deriv, error = update(w,b,train_x, train_y, alpha)
        
    w = params['w']
    b = params['b']
    
    # menghitung rata-rata
    train_error = np.mean(train_error)
    validation_error = np.mean(validation_error)
    train_accuracy = np.mean(train_accuracy)
    validation_accuracy = np.mean(validation_accuracy)
    train_prediction = np.mean(train_prediction)
    validation_prediction = np.mean(validation_prediction)
    
    # menyimpan hasil
    result = {"train_error": train_error,
              "validation_error": validation_error,
              "train_prediction": train_prediction,
              "validation_prediction": validation_prediction,
              "train_accuracy": train_accuracy,
              "validation_accuracy": validation_accuracy,
              "w": w,
              "b": b,
              "train_y": train_y,
              "validation_y": validation_y}
        
    return result  
    
train_error = []
validation_error = []
train_accuracy = []
validation_accuracy = []

w,b = initialize(4)
alpha = 0.1

# n_epoch = 300
for i in range(300):
    result = model(data,w,b,alpha)
    w = result['w']
    b = result['b']
    train_error.append(result['train_error'])
    validation_error.append(result['validation_error'])
    train_accuracy.append(result['train_accuracy'])
    validation_accuracy.append(result['validation_accuracy'])
    
# Visualisasi Error Function
plt.plot(train_error)
plt.plot(validation_error)
plt.xlabel('Iterasi')
plt.ylabel('Error')
plt.legend(labels=['training','validation'])
plt.show()

# Visualisasi Akurasi
plt.plot(train_accuracy)
plt.plot(validation_accuracy)
plt.xlabel('Iterasi')
plt.ylabel('Accuracy')
plt.show()
plt.legend(labels=['training','validation'])
