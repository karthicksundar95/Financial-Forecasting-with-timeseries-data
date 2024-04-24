""" Module to perform automated forecasting experiment and reporting"""

# Necessary packages are imported
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from pmdarima.arima import auto_arima, ADFTest


class Forecast:
    """
    Class designed to execute automated ML for forecasting experiment
    """
    def __init__(self, data=None, colname='Adj Close',
                 ma_window_size=3, wma_weights=[0.1, 0.4, 0.5], stock_code=None,
                 startdate=None, enddate=None):
        """
        Initialise all the class self variables
        :param data: python dataframe with time series data
        :param colname: name of column to work on for univariate time series
        :param ma_window_size: order size for moving average
        :param wma_weights: list of weights for weighted moving average
        :param stock_code: yahoo stock code for the stock to pull data from yfinance
        :param startdate: start data of data to pull from
        :param enddate: end data of the data range to pull from
        """
        self.ma_n = ma_window_size
        self.wa_wgts = wma_weights
        self.ewa_span = ma_window_size
        self.ewa_min_period = self.ewa_span + 1
        self.stock_code = stock_code
        self.data = data
        self.max_epochs = 200
        self.batch_size = 32
        if stock_code and startdate and enddate:
            self.start_date = startdate
            self.end_date = enddate
            self.ts_col_name = colname
        elif data and data[colname]:
            self.data = data
            self.ts_col_name = colname
        else:
            raise Exception("Expected value for startdate and enddate if stock_code provided! [OR]\
                            Expectecd data and the column name to work on")
        self.result = {}
        self.writer = pd.ExcelWriter("fin_forecast_report.xlsx", engine='xlsxwriter')

    def fetch_data(self):
        """
        To capture the stock data from yahoo finance website
        :return: pandas dataframe with the data from the given time range
        """
        data = yf.download(self.stock_code, self.start_date, self.end_date)
        self.data = data
        self.data.to_excel(self.writer, sheet_name='fetched_data')
        workbook = self.writer.book
        worksheet = self.writer.sheets['fetched_data']
        format_ = workbook.add_format({'text_wrap': True})
        worksheet.set_column('A:K', None, format_)
        worksheet.autofit()

    def moving_average(self):
        """
        Build moving average model on the given data for the given window size
        :return: prediction from moving average and
        comparative plot with actual data and moving average data
        """
        self.data['MA_forecast'] = self.data[self.ts_col_name].shift(1)\
            .rolling(window=self.ma_n,min_periods=None).mean()
        self.result['Moving Average'] = {}
        self.result['Moving Average']['Config'] = f"window size = {self.ma_n}"
        self.result['Moving Average']['MAE'] = abs(self.data['Adj Close'] - self.data['MA_forecast']).mean()

        self.data.to_excel(self.writer, sheet_name='MA_forecast')
        plt.figure()
        self.data['Adj Close'].plot(figsize=(50, 20), lw=4, legend=True)
        self.data['MA_forecast'].plot(legend=True)
        plt.savefig('ma_forecast.png')
        workbook = self.writer.book
        worksheet = self.writer.sheets['MA_forecast']
        format_ = workbook.add_format({'text_wrap': True})
        worksheet.set_column('A:K', None, format_)
        worksheet.autofit()
        worksheet.insert_image('K3', 'ma_forecast.png', {'x_scale': 0.25, 'y_scale': 0.25})

    def weighted_moving_average(self):
        """
        To build weighted moving average with default weights or user provided weights
        :return: Predictions from the model and comparative plot with actual and predicted data
        """
        self.data['MA_weighted_forecast'] = self.data[self.ts_col_name].shift(1) \
            .rolling(window=self.ma_n, min_periods=None) \
            .apply(lambda seq: np.average(seq, weights=self.wa_wgts))
        self.result['Weighted Moving Average'] = {}
        self.result['Weighted Moving Average']['Config'] = f"Window size = {self.ma_n}; Weights = {self.wa_wgts}"
        self.result['Weighted Moving Average']['MAE'] = abs(
            self.data[self.ts_col_name] - self.data['MA_weighted_forecast']).mean()

        self.data.to_excel(self.writer, sheet_name='Weighted_MA_forecast')
        plt.figure()
        self.data[self.ts_col_name].plot(figsize=(50, 20), lw=4, legend=True)
        self.data['MA_weighted_forecast'].plot(legend=True)
        plt.savefig('ma_weighted_forecast.png')
        workbook = self.writer.book
        worksheet = self.writer.sheets['Weighted_MA_forecast']
        format_ = workbook.add_format({'text_wrap': True})
        worksheet.set_column('A:K', None, format_)
        worksheet.autofit()
        worksheet.insert_image('K3', 'ma_weighted_forecast.png', {'x_scale': 0.25, 'y_scale': 0.25})

    def exponential_moving_average(self):
        """
        To build a exponential moving average model for the provided or default window size
        :return: Predictions from the model and comparative plot of actual and predicted data
        """
        self.data['EWMA_forecast'] = self.data[self.ts_col_name].ewm(span=self.ewa_span, adjust=True,
                                                                      min_periods=self.ewa_min_period).mean().to_list()
        self.result['Exp. Weighted Moving Average'] = {}
        self.result['Exp. Weighted Moving Average']['Config'] = f"Window size = {self.ewa_span}"
        self.result['Exp. Weighted Moving Average']['MAE'] = abs(
            self.data[self.ts_col_name] - self.data['EWMA_forecast']).mean()

        self.data.to_excel(self.writer, sheet_name='EWMA_forecast')
        plt.figure()
        self.data[self.ts_col_name].plot(figsize=(50, 20), lw=4, legend=True)
        self.data['EWMA_forecast'].plot(legend=True)
        plt.savefig('EWMA_forecast.png')
        workbook = self.writer.book
        worksheet = self.writer.sheets['EWMA_forecast']
        format_ = workbook.add_format({'text_wrap': True})
        worksheet.set_column('A:K', None, format_)
        worksheet.autofit()
        worksheet.insert_image('K3', 'EWMA_forecast.png', {'x_scale': 0.25, 'y_scale': 0.25})

    def auto_arima(self):
        """
        To build a ARIMA model in an automated grid search manner
        :return: Predictions from auto ARIMA and comparative plot with actual and predicted data
        """
        adf_test = ADFTest(alpha=0.05)
        adf_test.should_diff(self.data[self.ts_col_name])
        arima_model = auto_arima(self.data[self.ts_col_name], start_p=0,
                                 max_p=10, start_q=0, max_q=10, d=0, max_d=10,
                                 start_Q=0, max_Q=10, start_P=0, max_P=10, seasonal=True,
                                 trace=True, suppress_warnings=True, stepwise=True, random_state=42, n_fits=100)
        self.data['ARIMA_forecast'] = arima_model.predict_in_sample(self.data[self.ts_col_name])
        self.result['ARIMA'] = {}
        self.result['ARIMA']['Config'] = arima_model.get_params()
        self.result['ARIMA']['MAE'] = abs(self.data[self.ts_col_name]
                                          - self.data['ARIMA_forecast']).mean()

        self.data.to_excel(self.writer, sheet_name='ARIMA')
        plt.figure()
        self.data[self.ts_col_name].plot(figsize=(50, 20), lw=4, legend=True)
        arima_model.predict_in_sample(self.data[self.ts_col_name]).plot(legend=True)
        plt.savefig('ARIMA_forecast.png')
        workbook = self.writer.book
        worksheet = self.writer.sheets['ARIMA']
        format_ = workbook.add_format({'text_wrap': True})
        worksheet.set_column('A:G', None, format_)
        worksheet.autofit()
        worksheet.insert_image('M3', 'ARIMA_forecast.png', {'x_scale': 0.25, 'y_scale': 0.25})

    def rnn_data_prep(self, df):
        """
        To transform the pandas dataframe to suit the data format for tensorflow
        :param df: pandas dataframe
        :return: new dataframe in a tensorflow format
        """
        x_window = []
        y_price = []

        for day in range(5, df.shape[0]):
            # print(day)
            row = df[day - 5:day, 0]
            x_window.append(row)

            y = df[day, 0]
            y_price.append(y)
        x, y = np.array(x_window), np.array(y_price)
        x = x.reshape(x.shape[0], x.shape[1], 1)

        return x, y

    def rnn_architecture(self, df):
        """
        Defining the architecture of the RNN model
        :param df: dataframe shape to set teh architecture
        :return:
        """
        loss = 'mean_squared_error'
        neurons = 50
        hidden_layers = 2
        dense_layers = 1

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.SimpleRNN(neurons,
                                            activation='tanh',
                                            return_sequences=True,
                                            input_shape=(df.shape[1], 1)))
        model.add(tf.keras.layers.Dropout(0.2))

        for i in range(hidden_layers):
            model.add(tf.keras.layers.SimpleRNN(neurons,
                                                activation='tanh',
                                                return_sequences=True))
            model.add(tf.keras.layers.Dropout(0.2))

        # return sequense changed to false
        model.add(tf.keras.layers.SimpleRNN(neurons,
                                            activation='tanh',
                                            return_sequences=False))
        model.add(tf.keras.layers.Dropout(0.2))

        for i in range(dense_layers):
            model.add(tf.keras.layers.Dense(units=neurons,
                                            activation='tanh'))

        # Output
        model.add(tf.keras.layers.Dense(units=1))
        model.summary()
        opt = tf.keras.optimizers.Adam()
        model.compile(optimizer=opt,
                      loss=loss)

        return model

    def min_max_scaler(self, df):
        """
        To scale the data to single scale range
        :param df: pandas dataframe with the time series data
        :return: scaled dataframe time series data
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(np.array(df).reshape(-1, 1))
        return scaler, data_scaled

    def rnn(self):
        """
        To build a simple RNN model to train and predict the time series data
        :return: forecasted data with comparative plot with actual data
        """
        scaler, data_scaled = self.min_max_scaler(self.data[self.ts_col_name])
        x, y = self.rnn_data_prep(data_scaled)
        model = self.rnn_architecture(x)

        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              mode='min', patience=200, min_delta=0.00001,
                                              verbose=1)
        history = model.fit(x, y, epochs=self.max_epochs,
                            batch_size=self.batch_size, validation_split=0.2,
                            callbacks=[es])
        pred = model.predict(x=x).tolist()
        output = scaler.inverse_transform(pred)
        org_vals = scaler.inverse_transform(y.reshape(-1, 1))

        self.data['RNN_forecast'] = None
        self.data.loc[5:, 'RNN_forecast'] = output
        self.result['RNN'] = {}
        self.result['RNN']['Config'] = self.summary(model)
        self.result['RNN']['MAE'] = abs(self.data[self.ts_col_name]
                                        - self.data['RNN_forecast']).mean()
        sheetname = f'RNN_{self.max_epochs if es.stopped_epoch == 0 else es.stopped_epoch}_epochs'
        self.data.to_excel(self.writer, sheet_name=sheetname)

        plt.figure(figsize=(40, 30), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(org_vals, color="Green", label="Org value")
        plt.plot(output, color="orange", label="Predicted")
        plt.legend()
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.grid(True)
        plt.savefig('RNN_forecast.png')
        workbook = self.writer.book
        worksheet = self.writer.sheets[sheetname]
        format_ = workbook.add_format({'text_wrap': True})
        worksheet.set_column('A:K', None, format_)
        worksheet.autofit()
        worksheet = self.writer.sheets[sheetname]
        worksheet.insert_image('K3', 'RNN_forecast.png', {'x_scale': 0.25, 'y_scale': 0.25})

    def lstm_architecture(self):
        """
        To define the architecture of the LSTM network
        :return: compiled architecture of LSTM network
        """
        loss = 'mean_squared_error'
        neurons = 50
        hidden_layers = 2
        dense_layers = 1

        model = tf.keras.Sequential()

        model.add(tf.keras.layers.LSTM(neurons, input_shape=(None, 1)))
        model.add(tf.keras.layers.Dense(units=1))
        opt = tf.keras.optimizers.Adam()
        model.compile(optimizer=opt,
                      loss=loss)

        return model

    def lstm(self):
        """
        To build a LSTM model with the proposed architecture
        :return: forecasted values from the model
        and comparative plots with actual and forecasted data
        """
        scaler, data_scaled = self.min_max_scaler(self.data[self.ts_col_name])
        x, y = self.rnn_data_prep(data_scaled)
        model = self.lstm_architecture()

        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                              patience=200, min_delta=0.00001,
                                              verbose=1)
        history = model.fit(x, y, epochs=self.max_epochs,
                            batch_size=self.batch_size, validation_split=0.2,
                            callbacks=[es])
        pred = model.predict(x=x).tolist()
        output = scaler.inverse_transform(pred)
        org_vals = scaler.inverse_transform(y.reshape(-1, 1))

        self.data['LSTM_forecast'] = None
        self.data.loc[5:, 'LSTM_forecast'] = output
        self.result['LSTM'] = {}
        self.result['LSTM']['Config'] = self.summary(model)
        self.result['LSTM']['MAE'] = abs(self.data[self.ts_col_name] -
                                         self.data['LSTM_forecast']).mean()

        sheetname = 'LSTM_{}_epochs'.format(self.max_epochs
                                            if es.stopped_epoch == 0 else es.stopped_epoch)
        self.data.to_excel(self.writer, sheet_name=sheetname)

        plt.figure(figsize=(40, 30), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(org_vals, color="Green", label="Org value")
        plt.plot(output, color="orange", label="Predicted")
        plt.legend()
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.grid(True)
        plt.savefig('LSTM_forecast.png')
        workbook = self.writer.book
        worksheet = self.writer.sheets[sheetname]
        format_ = workbook.add_format({'text_wrap': True})
        worksheet.set_column('A:K', None, format_)
        worksheet.autofit()
        worksheet = self.writer.sheets[sheetname]
        worksheet.insert_image('K3', 'LSTM_forecast.png', {'x_scale': 0.25, 'y_scale': 0.25})

    def summary(self, model: tf.keras.Model):
        """
        Textual summary of the architecture and parameters from tensorflow model
        :param model: tensorflow model
        :return: string summary of the model
        """
        summary = []
        model.summary(print_fn=lambda x: summary.append(x))
        return '\n'.join(summary)

    def run(self):
        """
        Main control method to run all the algorithms in an automated fashion
        :return: Final summary sheet and saving the entire report
        """
        self.fetch_data()
        print("Finished fetching data..")
        self.moving_average()
        print("Finished forecasting using moving average...")
        self.weighted_moving_average()
        print("Finished forecasting using WMA...")
        self.exponential_moving_average()
        print("Finished forecasting using EWMA...")
        self.auto_arima()
        print("Finished forecasting using AUTO ARIMA...")
        self.rnn()
        print("Finished forecasting using RNN...")
        self.lstm()
        print("Finished forecasting using LSTM...")
        print("Results are available at fin_forecast_report.xlsx NOW!!")

        pd.DataFrame(self.result).to_excel(self.writer, sheet_name='Summary')
        workbook = self.writer.book
        worksheet = self.writer.sheets['Summary']
        format_ = workbook.add_format({'text_wrap': True})
        format_.set_align('vcenter')
        # worksheet.autofit()
        worksheet.set_column('A:G', None, format_)
        worksheet.autofit()
        self.writer.save()
        return pd.DataFrame(self.result)
