import csv
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.datasets import cifar10
from sklearn import datasets
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn
from sklearn.metrics import confusion_matrix
 



letters=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
directories=["dataset1", "dataset2"]

#directories=["dataset1", "dataset2"]

data=[]
label=[]



for dir in directories:
    for i in letters:
        cur_data=pd.read_csv('/Users/jx/Desktop/Col/ISEF/isef_new_program/my_dataset/'+dir+'/'+i+'.csv')
        number_index=letters.index(i)
        for se in cur_data.values:
            data.append(se)
            label.append(number_index)




model = Sequential()

model.add(Conv1D(filters=32, kernel_size=5, strides=1, activation='relu'))
model.add(Conv1D(filters=32, kernel_size=2, strides=1, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())
model.add(Dense(units=64,activation='relu'))
model.add(Dense(units=26, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x=data
y=label
x_train,x_test,y_train,y_test = train_test_split(np.array(x).reshape(-1,63,1),y,test_size=0.1)
labels = to_categorical(y_train)

history = model.fit(x_train, labels, epochs=10, batch_size=64, validation_split=0.1)

y_cat_test= to_categorical(y_test)

loss_and_metrics = model.evaluate(x_test, y_cat_test)

model.summary()

#model.save("model_1.h5")

print(loss_and_metrics)

plt.subplot(2,1,1)
plt.title("Loss & Accuracy vs Epoch")
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["Loss", "Validation Loss"])

plt.subplot(2,1,2)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.legend(["Accuracy", "Validation Accuracy"])
plt.show()


y_predict=model.predict(x_test)

y_pred=[np.argmax(i) for i in y_predict]

cn=confusion_matrix(y_test, y_pred)

seaborn.heatmap(cn)
plt.xlabel("Predict")
plt.ylabel("True")
plt.show()