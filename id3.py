import streamlit as st
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

def main():
    st.title("Decision Tree Classifier")

    # Get the current directory path
    current_dir = os.path.dirname(__file__)

    # Construct the full path to the CSV file
    csv_file_path = os.path.join(current_dir, 'sampledata1.csv')

    # Load training data from CSV
    @st.cache_data  # Cache data to avoid loading it multiple times
    def load_data(file_path):
        return pd.read_csv(file_path)

    df = load_data(csv_file_path)

    # Separate features and labels
    X = df.drop('Label', axis=1)
    y = df['Label']

    # One-hot encode categorical features
    categorical_features = ['Outlook', 'Temperature', 'Humidity', 'Windy']
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(), categorical_features)],
        remainder='passthrough'
    )
    X_encoded = preprocessor.fit_transform(X)

    # Initialize and fit the decision tree model
    model = DecisionTreeClassifier(criterion='entropy')  # ID3 uses entropy
    model.fit(X_encoded, y)

    st.write("Enter new sample attributes to classify:")
    attributes = []
    for col in X.columns:
        attributes.append(st.text_input(f"{col}:"))

    if st.button("Classify"):
        # Encode the new sample using the same encoder
        new_sample = preprocessor.transform(pd.DataFrame([attributes], columns=X.columns))
        prediction = model.predict(new_sample)
        st.write("Predicted class for new sample:", prediction[0])

if __name__ == "__main__":
    main()
