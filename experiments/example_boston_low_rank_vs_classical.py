import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.elm.elms import ClassicELM, LowRankELM


X, y = load_boston(return_X_y=True)
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2)
y_train_raw = np.expand_dims(y_train_raw, -1)
y_test_raw = np.expand_dims(y_test_raw, -1)

# Scale the Inputs
X_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train_raw)
X_test = X_scaler.transform(X_test_raw)

# Scale the outputs
y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train_raw)
y_test = y_scaler.transform(y_test_raw)

# "Train" the standard ELM
standard_elm = ClassicELM(hidden_neurons_numbers=[200, 200], activation="relu", input_shape=X_train[0].shape)
standard_elm.fit(X_train, y_train)
standard_elm_test_pred = standard_elm.predict(X_test)
standard_elm_test_pred_backscaled = y_scaler.inverse_transform(standard_elm_test_pred)

# "Train" the low rank ELM
low_rank_elm = LowRankELM([200, 200], [200, 20], activation="relu", input_shape=X_train[0].shape)
low_rank_elm.fit(X_train, y_train)
low_rank_elm_test_pred = low_rank_elm.predict(X_test)
low_rank_elm_test_pred_backscaled = y_scaler.inverse_transform(low_rank_elm_test_pred)

# Print the results
print(f"Standard ELM Mean Squared Test Error: {np.mean(np.square(standard_elm_test_pred_backscaled.flatten() - y_test_raw.flatten()))}")
print(f"Standard ELM number of parameters (without output layer): {standard_elm.keras_model.count_params()}")
print(f"Low Rank ELM Mean Squared Test Error: {np.mean(np.square(low_rank_elm_test_pred_backscaled.flatten() - y_test_raw.flatten()))}")
print(f"Low Rank ELM number of parameters (without output layer): {low_rank_elm.keras_model.count_params()}")