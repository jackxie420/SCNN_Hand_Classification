from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.datasets import cifar10
from sklearn import datasets
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
 

model = Sequential()

data=datasets.load_iris()

#model.add(Flatten())
model.add(Dense(units=64,activation='relu'))
model.add(Dense(units=3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

#(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x=data.data
y=data.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)
labels = to_categorical(y_train)

history = model.fit(x_train, labels, epochs=10, batch_size=64, validation_split=0.1)

model.summary()

loss_and_metrics = model.evaluate(x_test, to_categorical(y_test))

print(loss_and_metrics)
plt.subplot(2,1,1)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])


plt.subplot(2,1,2)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.show()



