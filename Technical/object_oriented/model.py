import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

class LstmModel:
    def __init__(self, ticker):
        self.stock_symbol = ticker
        self.data = None
        self.normalizer = MinMaxScaler(feature_range=(0, 1))
        self.ds_scaled = None
        self.train_size = None
        self.test_size = None
        self.ds_train = None
        self.ds_test = None
        self.time_stamp = 100
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None

    def download_data(self):
        self.data = yf.download(tickers=self.stock_symbol, period='5y', interval='1d')

    def preprocess_data(self):
        self.ds_scaled = self.normalizer.fit_transform(np.array(self.data['Close']).reshape(-1, 1))
        self.train_size = int(len(self.ds_scaled) * 0.70)
        self.test_size = len(self.ds_scaled) - self.train_size
        self.ds_train, self.ds_test = self.ds_scaled[0:self.train_size, :], self.ds_scaled[self.train_size:len(self.ds_scaled), :1]
        self.X_train, self.y_train = self.create_ds(self.ds_train, self.time_stamp)
        self.X_test, self.y_test = self.create_ds(self.ds_test, self.time_stamp)
        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1)

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(self.X_train.shape[1], 1)))
        self.model.add(LSTM(units=50, return_sequences=True))
        self.model.add(LSTM(units=50))
        self.model.add(Dense(units=1, activation='linear'))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def train_model(self):
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=100, batch_size=64)

    def predict_future(self):
        train_predict = self.model.predict(self.X_train)
        test_predict = self.model.predict(self.X_test)
        train_predict = self.normalizer.inverse_transform(train_predict)
        test_predict = self.normalizer.inverse_transform(test_predict)
        test = np.vstack((train_predict, test_predict))

        fut_inp = self.ds_test[278:]
        fut_inp = fut_inp.reshape(1, -1)
        tmp_inp = list(fut_inp)
        tmp_inp = tmp_inp[0].tolist()

        lst_output = []
        n_steps = 100
        i = 0
        while i < 30:
            if len(tmp_inp) > 100:
                fut_inp = np.array(tmp_inp[1:])
                fut_inp = fut_inp.reshape(1, -1)
                fut_inp = fut_inp.reshape((1, n_steps, 1))
                yhat = self.model.predict(fut_inp, verbose=0)
                tmp_inp.extend(yhat[0].tolist())
                tmp_inp = tmp_inp[1:]
                lst_output.extend(yhat.tolist())
                i += 1
            else:
                fut_inp = fut_inp.reshape((1, n_steps, 1))
                yhat = self.model.predict(fut_inp, verbose=0)
                tmp_inp.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
                i += 1

        ds_new = self.ds_scaled.tolist()
        ds_new.extend(lst_output)
        final_graph = self.normalizer.inverse_transform(ds_new).tolist()

        # plt.plot(final_graph,)
        # plt.ylabel("Price")
        # plt.xlabel("Time")
        # plt.title("{0} prediction of next month open".format(self.stock_symbol))
        # plt.axhline(y=final_graph[len(final_graph)-1], color = 'red', linestyle = ':', label = 'NEXT 30D: {0}'.format(round(float(*final_graph[len(final_graph)-1]),2)))
        # plt.legend()
        # plt.show() #Should comment this line for the final model
        return round(float(*final_graph[len(final_graph)-1]),2)

    @staticmethod
    def create_ds(dataset, step):
        Xtrain, Ytrain = [], []
        for i in range(len(dataset) - step - 1):
            a = dataset[i:(i + step), 0]
            Xtrain.append(a)
            Ytrain.append(dataset[i + step, 0])
        return np.array(Xtrain), np.array(Ytrain)


