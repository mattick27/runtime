from keras.datasets import mnist
from keras.models import Sequential,Model
from keras.layers import Input, Embedding, LSTM, Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
import keras
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0],1,28,28)
X_test = X_test.reshape(X_test.shape[0],1,28,28)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
model = Sequential()

mainInput = Input(shape=(1,28,28),name='mainInput',dtype='float32')
filtered_1 = Conv2D(32,(3,3),padding='same',name='filtered_1')(mainInput)
x1 = MaxPooling2D((2, 2), strides=(2, 2), name='block_pool1',dim_ordering="th")(filtered_1)

filtered_2 = Conv2D(64,(3,3),padding='same',name='filtered_2')(mainInput)
x2 = MaxPooling2D((2, 2), strides=(2, 2), name='block_pool2',dim_ordering="th")(filtered_2)

filtered_3 = Conv2D(128,(3,3),padding='same',name='filtered_3')(mainInput)
x3 = MaxPooling2D((2, 2), strides=(2, 2), name='block_pool3',dim_ordering="th")(filtered_3)

filtered_4 = Conv2D(32,(5,5),padding='same',name='filtered_4')(mainInput)
x4 = MaxPooling2D((2, 2), strides=(2, 2), name='block_pool4',dim_ordering="th")(filtered_4)

x = keras.layers.concatenate([x1,x2,x3,x4])
flat = Flatten()(x)
x = Dense(32,activation='relu')(flat)
main_output = Dense(1, activation='softmax', name='main_output')(x)
print(main_output)
model = Model(inputs=[mainInput], outputs=[main_output])
model.compile(optimizer='rmsprop', loss='binary_crossentropy')
model.fit(X_train, y_train,epochs=10, batch_size=32)
score = model.evaluate(X_test,y_test,batch_size=32)
print(model.output_shape)
print(score)