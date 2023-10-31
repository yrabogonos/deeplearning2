
import streamlit as st
from streamlit.logger import get_logger
import pandas as pd
import matplotlib.pyplot as plt 
import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow

LOGGER = get_logger(__name__)


def run():

  housing = fetch_california_housing()
  X = pd.DataFrame(housing.data, columns=housing.feature_names)
  y = pd.DataFrame(housing.target, columns=housing.target_names)
  housing_data = pd.concat([X, y], axis=1)
  
  # Розділення даних
  x_train_full, x_test, y_train_full, y_test = train_test_split(housing_data, housing.target, random_state=42)
  x_train, x_valid, y_train, y_valid = train_test_split(x_train_full, y_train_full, random_state=42)
  
  scaler = StandardScaler()
  x_train_scaled = scaler.fit_transform(x_train)
  x_valid_scaled = scaler.transform(x_valid)
  x_test_scaled = scaler.transform(x_test)
  
  # Створення моделей
  def create_model(model_type):
      if model_type == 'a':
          model = keras.models.Sequential()
          model.add(keras.layers.Dense(30, activation='relu', input_shape=x_train_scaled.shape[1:]))
          model.add(keras.layers.Dense(20, activation='relu'))
          model.add(keras.layers.Dense(10, activation='relu'))
          model.add(keras.layers.Dense(1))
          return model
      elif model_type == 'b':
        
        input_deep = keras.layers.Input(shape=(x_train_scaled.shape[1:]))
        hidden1 = keras.layers.Dense(30, activation='relu')(input_deep)
        hidden2 = keras.layers.Dense(20, activation='relu')(hidden1)
        input_short = keras.layers.Input(shape=(x_train_scaled.shape[1:]))
        hidden3 = keras.layers.Dense(10, activation='relu')(input_short)
        concatenated = keras.layers.concatenate([hidden2, hidden3])
        output = keras.layers.Dense(1)(concatenated)
        return keras.models.Model(inputs=[input_deep, input_short], outputs=output)
      elif model_type == 'c':
          # Додайте код для створення моделі "c"
          pass
      elif model_type == 'd':
          # Додайте код для створення моделі "d"
          pass
      return model
  
  # Вибір моделі
  model_type = st.selectbox("Виберіть модель", ['a', 'b', 'c', 'd'])
  model = create_model(model_type)
  model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(lr=1e-3))
  #keras.utils.plot_model(model, 'regression_model.png', show_shapes=True)
  #----------------
  x_train_deep = x_train_scaled
  x_train_short = x_train_scaled
  x_valid_a = x_valid_scaled
  x_valid_b = x_valid_scaled
  x_test_a = x_test_scaled
  x_test_b = x_test_scaled
  y_test_a = y_test
  y_test_b = y_test
  #------------
  # Виведення моделі
  #st.subheader("Схема моделі")
  #st.image("regression_model.png")  # Додайте URL або шлях до схеми моделі
  
  # Тренування моделі
  st.subheader("Тренування моделі")
  if model_type == 'a':
    history = model.fit(x_train_scaled, y_train, epochs=30, validation_data=(x_valid_scaled, y_valid))
  else:
    history = model.fit((x_train_deep, x_train_short), y_train, epochs=30,validation_data=((x_valid_a, x_valid_b), y_valid))
  
  st.line_chart(pd.DataFrame(history.history))

  #Error
  if model_type=='a':
    mse_test = model.evaluate(x_test_scaled, y_test)
  else:
    mse_test = model.evaluate((x_test_a, x_test_b), (y_test_a, y_test_b))
  st.write("Mean Squared Error on Test Set:", mse_test)
  
  
  # Передбачення для обраного елемента
  st.subheader("Передбачення для обраного елемента")
  element_index = st.number_input("Введіть індекс елемента з тестового набору", min_value=0, max_value=len(x_test) - 1, value=0)
  x_subset = x_test_scaled[element_index].reshape(1, -1)
  y_expected = y_test[element_index]
  if model_type == 'a':
    y_predicted = model.predict(x_subset)
  else:
    y_predicted = model.predict((x_subset,x_subset))
  
  st.write(f"Елемент {element_index}:  Очікуване={y_expected}, Передбачене={y_predicted[0][0]}")


if __name__ == "__main__":
    run()
