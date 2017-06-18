import keras
import numpy as np

from keras.layers import LSTM, Embedding, Flatten, Dense, TimeDistributed
from keras.models import  Sequential
from keras.preprocessing import sequence

model = Sequential()

model.add(LSTM(128, input_shape=(10,1), return_sequences=True))

model.add(LSTM(64, return_sequences=True))

model.add(LSTM(32, return_sequences=True))

model.add(TimeDistributed(Dense(1)))

model.compile(optimizer='adam', loss='mse')

model.summary()


def generate_data(num_seq, means, variances):

    x = []
    y = []
    for i in range(num_seq):
        mean = np.random.uniform(-11,11)
        std =  np.random.uniform(0,11)
        x.append(np.random.normal(mean, std, (10,1)))
        y.append(np.full((10, 1), mean))

    x = np.vstack(x).reshape(-1, 10, 1)
    y = np.vstack(y).reshape(-1,10,1)

    print(x.shape, y.shape)
    return x, y



def generate_example(l):
    mean = np.random.uniform(-11,11, 20)
    std = np.random.uniform(0,11, 10)
    x = np.random.normal(mean, std, (10, 1))
    y = np.full((10, 1), mean)
    yield (x, y)



x,y = generate_data(20000, means=np.random.uniform(-11,11, 20), variances=np.random.uniform(0,11, 10))

model.fit(x,y, nb_epoch=600 , shuffle=True,callbacks=[keras.callbacks.ModelCheckpoint(save_best_only=False, period=10, filepath='/Users/vlasteli/PycharmProjects/artificial_neural_networks/gen_gauss_player')])








