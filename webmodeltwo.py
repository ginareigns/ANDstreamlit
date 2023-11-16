import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


# Load your Scikit-learn model
model = joblib.load('random_forest_model.joblib')

# ... [rest of your preprocess function] ...
def preprocess(df):
    # Fill missing 'attack_cat' with 'Normal'
    df['attack_cat'].fillna('Normal', inplace=True)
    
    # Filter out 'Normal' traffic and consolidate categories
    df = df[df['attack_cat'] != 'Normal']
    df['attack_cat'] = df['attack_cat'].apply(lambda x: x.strip())
    
    # Consolidate categories in 'attack_cat'
    replacements = {
        'Exploits': 'DoS', 'Fuzzers': 'DoS', 'Reconnaissance': 'Port Scan',
        'Analysis': 'Port Scan', 'Backdoors': 'Privilege Escalation',
        'Backdoor': 'Privilege Escalation', 'Shellcode': 'Privilege Escalation',
        'Worms': 'Privilege Escalation'
    }
    df['attack_cat'].replace(replacements, inplace=True)
    
    # Select relevant features and create dummy variables
    X = df[['proto', 'Spkts', 'Dpkts', 'tcprtt', 'state', 'dur', 'sbytes', 'dbytes', 'ct_srv_src', 'ct_srv_dst']]
    X = pd.get_dummies(X, columns=['proto', 'state'])
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled


# Streamlit interface code
st.title('Network Traffic Anomaly Detection System')
st.write('Upload network traffic data and detect anomalies.')

# File uploader allows user to add their own CSV
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")


if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write('Data successfully loaded!')
    
    if st.button('Detect Anomalies'):
        # Preprocess the uploaded data
        preprocessed_data = preprocess(data)
        
        # Use the model to predict anomalies
        predictions = model.predict(preprocessed_data)

        # Flatten the predictions if they're not already 1D
        if predictions.ndim > 1:
            predictions = np.argmax(predictions, axis=1)

        # Create a DataFrame from the 1D predictions array
        predictions_df = pd.DataFrame(predictions, columns=['Predictions'])

        # Concatenate the original data with the predictions
        results = pd.concat([data.reset_index(drop=True), predictions_df], axis=1)
        
        # Display the data along with predictions
        st.warning('Anomalies detected:')
        st.dataframe(results)

        # Ensure the 'Predictions' column exists before plotting
        if 'Predictions' in results.columns:
            # Visualize the prediction distribution with a bar chart
            st.subheader('Prediction Distribution:')
            fig, ax = plt.subplots()
            sns.countplot(x='Predictions', data=results, ax=ax)
            ax.set_title('Frequency of Predicted Classes')
            st.pyplot(fig)
        else:
            st.error('The "Predictions" column does not exist in the DataFrame.')

else:
    st.warning('Please upload a CSV file to get started.')
