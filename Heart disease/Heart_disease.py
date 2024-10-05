import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import models, layers
import matplotlib.pyplot as plt

# Title of the app
st.title('Heart Disease Prediction')

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('heart.csv')
    return data

data = load_data()

# Display data if checkbox is selected
if st.checkbox('Show raw data'):
    st.subheader('Raw Dataset')
    st.write(data.head())

# Correlation matrix display
if st.checkbox('Show correlation with target'):
    corr_matrix = data.corr()
    st.write(corr_matrix['target'].sort_values(ascending=False))

# Splitting features and target
Features = data.iloc[:, :-1]
Target = data.iloc[:, -1]

# Train-test split
train_Features, test_Features, train_target, test_target = train_test_split(Features, Target, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train = scaler.fit_transform(train_Features)
X_test = scaler.transform(test_Features)

# Define Keras model
def build_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(train_Features.shape[1],)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model()

# Train the model
if st.button('Train Model'):
    st.write('Training the model...')
    history = model.fit(X_train, train_target, epochs=50, batch_size=10, validation_split=0.2, verbose=0)
    st.write('Training complete!')

    # Plot training and validation accuracy
    st.subheader('Training and Validation Accuracy')
    acc_values = history.history['accuracy']
    val_acc_values = history.history['val_accuracy']
    epochs = range(1, len(acc_values) + 1)

    plt.plot(epochs, acc_values, 'bo', label='Training Accuracy')
    plt.plot(epochs, val_acc_values, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    st.pyplot(plt.gcf())  # Display plot in Streamlit

# Predicting heart disease based on test set
if st.button('Evaluate Model'):
    loss, accuracy = model.evaluate(X_test, test_target)
    st.write(f'Test Loss: {loss:.4f}')
    st.write(f'Test Accuracy: {accuracy:.4f}')

    # Make predictions on test set
    predicted_probability = model.predict(X_test)
    
    # Show predictions
    st.subheader("Predictions on Test Data")
    predictions = ['Heart Disease' if prob > 0.5 else 'No Heart Disease' for prob in predicted_probability]
    st.write(predictions)

    # Compare with actual values
    comparison_df = pd.DataFrame({'Actual': test_target.values, 'Predicted': predictions})
    st.write(comparison_df.head())

