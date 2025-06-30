import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder

# --- Load data ---
st.title("Relationship Compatibility Predictor")
st.write("Fill out the questionnaire to see if you're a match!")

# Load the CSV (you can replace this with another method if deploying)
df = pd.read_csv("Assessment - Form Responses.csv")
df.drop(columns=["Timestamp", "Email Address"], inplace=True)

# Generate target
data = {
    'How spontaneous are you?': "Balanced",
    'Do you enjoy giving or receiving surprises?\n': "Love it",
    '\nHow important is music taste compatibility to you?\n': "Very important, must match mine",
    'How open are you to trying new things (food, travel, experiences)?': "Mostly open",
    'How much personal space do you need in a relationship?': "Moderate",
    'How emotionally expressive are you?': "Balanced",
    'How important is having similar long-term goals?': "Important",
    'What’s your preferred mode of communication?': "Face to Face",
    'What’s your ideal time to hang out?': "Evening",
    'What is your ideal weekend plan?': "Outdoor Adventures"
}

def fun(row):
    return sum(row.get(key) == value for key, value in data.items())

df['score'] = df.apply(fun, axis=1)
df['target'] = df['score'].apply(lambda x: "match" if x > 4 else "not match")
df.drop(columns="score", inplace=True)

X = df.drop('target', axis=1)
y = df['target']

# --- Column categories ---
nominal_cols = [
    'Do you enjoy giving or receiving surprises?\n',
    'What’s your preferred mode of communication?',
    'What is your ideal weekend plan?'
]

ordinal_cols = [
    'How spontaneous are you?',
    '\nHow important is music taste compatibility to you?\n',
    'How open are you to trying new things (food, travel, experiences)?',
    'How much personal space do you need in a relationship?',
    'How emotionally expressive are you?',
    'How important is having similar long-term goals?',
    'What’s your ideal time to hang out?'
]

ordinal_orders = [
    ['Very planned', 'Mostly planned', 'Balanced', 'Mostly spontaneous', 'Very spontaneous'],
    ['Doesn’t matter at all', 'Slightly important', 'Neutral / Okay either way', 'Important but not a deal-breaker', 'Very important, must match mine'],
    ['Not open at all', 'Slightly hesitant', 'Sometimes open', 'Mostly open', 'Very adventurous'],
    ['Very little', 'A little', 'Moderate', 'Quite a bit', 'A lot'],
    ['Very reserved', 'Slightly reserved', 'Balanced', 'Mostly expressive', 'Very expressive'],
    ['Not important', 'Slightly important', 'Neutral', 'Important', 'Very important'],
    ['Morning', 'Afternoon', 'Evening', 'Late night']
]

# --- Pipelines ---
ordinal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(categories=ordinal_orders))
])

nominal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('ord', ordinal_pipeline, ordinal_cols),
    ('nom', nominal_pipeline, nominal_cols)
])

X_processed = preprocessor.fit_transform(X)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --- Build and train model ---
model = keras.Sequential([
    keras.layers.Input(shape=(X_processed.shape[1],)),
    keras.layers.Dense(20, activation='relu', kernel_initializer=keras.initializers.RandomNormal()),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.LeakyReLU(),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_processed, y_encoded, epochs=10, validation_split=0.2, verbose=0)

# --- Streamlit form ---
user_input = {}

for col in ordinal_cols:
    user_input[col] = st.selectbox(col.strip(), ordinal_orders[ordinal_cols.index(col)])

for col in nominal_cols:
    unique_vals = df[col].dropna().unique().tolist()
    user_input[col] = st.selectbox(col.strip(), unique_vals)

if st.button("Predict Match"):
    input_df = pd.DataFrame([user_input])
    input_transformed = preprocessor.transform(input_df)
    prediction = model.predict(input_transformed)[0][0]
    result = "💖 Match" if prediction > 0.5 else "💔 Not Match"
    st.markdown(f"### Result: **{result}**")
