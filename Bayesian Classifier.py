import streamlit as st
import pandas as pd
import numpy as np

def label_encode(data):
    unique_values = np.unique(data)
    label_map = {val: idx for idx, val in enumerate(unique_values)}
    encoded_data = [label_map[val] for val in data]
    return encoded_data, label_map

def main():
    st.title("Tennis Play Prediction")

    # Create a DataFrame
    data = {
        'Outlook': ['sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy', 'overcast', 'sunny', 'sunny', 'rainy', 'sunny', 'overcast', 'overcast', 'rainy'],
        'Temperature': ['hot', 'hot', 'hot', 'mild', 'cool', 'cool', 'cool', 'mild', 'cool', 'mild', 'mild', 'mild', 'hot', 'mild'],
        'Humidity': ['high', 'high', 'high', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'high'],
        'Windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
        'PlayTennis': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
    }
    df = pd.DataFrame(data)

    st.write("The first 5 values of data are:")
    st.write(df.head())

    # Obtain Train data and Train output
    X = df.iloc[:,:-1]
    st.write("\nThe First 5 values of train data are:\n", X.head())

    y = df.iloc[:,-1]
    st.write("\nThe first 5 values of Train output are:\n", y.head())

    # Convert categorical variables to numbers 
    X['Outlook'], outlook_map = label_encode(X['Outlook'])
    X['Temperature'], temp_map = label_encode(X['Temperature'])
    X['Humidity'], humidity_map = label_encode(X['Humidity'])

    st.write("\nNow the Train data is :\n", X.head())

    # Convert target labels to numbers
    y, play_tennis_map = label_encode(y)
    st.write("\nNow the Train output is\n", y)

    # Convert y to NumPy array for indexing
    y = np.array(y)

    # Split data into train and test sets
    data_size = X.shape[0]
    train_size = int(0.8 * data_size)
    np.random.seed(42)  # For reproducibility
    indices = np.random.permutation(data_size)
    train_indices, test_indices = indices[:train_size], indices[train_size:]
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    # Train Gaussian Naive Bayes manually
    prior_probabilities = {}
    for label in np.unique(y_train):
        prior_probabilities[label] = np.sum(y_train == label) / len(y_train)

    likelihoods = {}
    for feature in X_train.columns:
        likelihoods[feature] = {}
        for label in np.unique(y_train):
            label_indices = np.where(y_train == label)[0]
            feature_values = X_train.iloc[label_indices][feature]  # Use iloc for integer-based indexing
            value_counts = np.bincount(feature_values)
            total_counts = np.sum(value_counts)
            likelihoods[feature][label] = {value: count / total_counts for value, count in enumerate(value_counts)}

    # Predict using Naive Bayes
    def predict(X):
        predictions = []
        for idx, row in X.iterrows():
            posterior_probabilities = {}
            for label in np.unique(y_train):
                posterior_probabilities[label] = prior_probabilities[label]
                for feature, value in row.items():  # Use items() instead of iteritems()
                    if value in likelihoods[feature][label]:
                        posterior_probabilities[label] *= likelihoods[feature][label][value]
                    else:
                        # Handle unseen feature values by assuming a small probability
                        posterior_probabilities[label] *= 1e-6
            predictions.append(max(posterior_probabilities, key=posterior_probabilities.get))
        return predictions

    # Evaluate accuracy
    y_pred = predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    st.write("Accuracy is:", accuracy)

if __name__ == "__main__":
    main()
