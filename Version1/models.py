import numpy as np
from keras.layers import LSTM, Dense
from keras.models import Sequential


def dense_model(x_train_dense, x_test_dense, y_train_dense, y_test_dense, num_hours=60, batch_size=1, epochs=5):
    """
    Neural network model that attempts to predict stock prices with several dense layers.

    :param epochs: number of epochs for which the model is trained - default of 5
    :param batch_size: size of one batch for training - default of 1
    :param num_hours: number of hours used to predict the next hour, default is 60
    :param x_train_dense: a np array with training inputs
    :param x_test_dense: a np array with testing inputs
    :param y_train_dense: a np array with expected training outputs
    :param y_test_dense: a np array with expected testing outputs
    :return: the trained model and its predictions on testing inputs
    """
    # reshape to 2D matrix with dims to fit the dense model: number of training examples x num_hours
    x_train_array = np.reshape(x_train_dense, (x_train_dense.shape[0], num_hours))
    x_test_array = np.reshape(x_test_dense, (x_test_dense.shape[0], num_hours))
    dense_model_layers = Sequential([
        Dense(50, activation='leaky_relu'),
        Dense(25, activation='leaky_relu'),
        Dense(1)
    ])
    dense_model_layers.compile(optimizer='adam', loss='mse')
    dense_model_layers.fit(x_train_array, y_train_dense, batch_size=batch_size, epochs=epochs)

    y_hats = dense_model_layers.predict(x_test_array)

    dense_rmse = np.sqrt(np.mean(y_hats - y_test_dense) ** 2)
    print(f"Dense Model RMSE: {dense_rmse}")
    return dense_model_layers, y_hats


def lstm_model(x_train_lstm, x_test_lstm, y_train_lstm, y_test_lstm, batch_size=1, epochs=5):
    """
    Neural network model that attempts to predict stock prices with LSTM and dense layers.

    :param epochs: number of epochs for which the model is trained - default of 5
    :param batch_size: size of one batch for training - default of 1
    :param x_train_lstm: a np array with training inputs
    :param x_test_lstm: a np array with testing inputs
    :param y_train_lstm: a np array with expected training outputs
    :param y_test_lstm: a np array with expected testing outputs
    :return: the trained model and the predictions it made on testing inputs
    """
    lstm_model_layers = Sequential([
        LSTM(units=64, return_sequences=True, input_shape=(x_train_lstm.shape[1], 1)),
        LSTM(units=64),
        Dense(25, activation='leaky_relu'),
        Dense(1)
    ])
    lstm_model_layers.summary()
    lstm_model_layers.compile(optimizer='adam', loss='mse')
    lstm_model_layers.fit(x_train_lstm, y_train_lstm, batch_size=batch_size, epochs=epochs)
    y_hats = lstm_model_layers.predict(x_test_lstm)

    lstm_rmse = np.sqrt(np.mean(y_hats - y_test_lstm) ** 2)
    print(f"LSTM Model RMSE: {lstm_rmse}")

    return lstm_model_layers, y_hats


def nn_trader_decision(model, latest_close, last_n_close_prices):
    """
    Uses neural network model to make a prediction and make a decision on buy/sell after comparing to latest close.

    :param last_n_close_prices: close prices for last n hours being used for prediction
    :param latest_close: latest close price
    :param model: the neural network model being used
    :return: True if the predicted price is higher than the latest price, False if not
    """
    prediction = 0
    # LSTM requires [0][0], Dense does not due to different matrix shapes
    if model._name == 'lstm':
        prediction = model.predict(last_n_close_prices)[0][0]
    elif model._name == 'dense':
        prediction = model.predict(last_n_close_prices)
    if prediction > latest_close:
        return True
    else:
        return False
