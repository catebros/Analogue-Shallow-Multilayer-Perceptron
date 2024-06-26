import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix

def get_model_accuracy(model, predictors_test, target_test):
    predictions = model.predict(predictors_test)  # get predictions

    # Convert outputs to binary predictions
    predictions_binary = [1 if p > 0.5 else 0 for p in predictions]

    # Calculate accuracy
    accuracy = (predictions_binary == target_test).mean()
    return f'Accuracy: {accuracy}'

def summary_df(df):
    # Print dataset info
    print(df.info())
    print(df.head())
    print(df.describe())
    print(df)

def create_csv(df, filename):
    df.to_csv(filename, index=False)
    
def normalize_model(train_x, test_x):
    scaler_x = MinMaxScaler(feature_range=(0, 5))
    
    train_x = scaler_x.fit_transform(train_x)
    test_x = scaler_x.transform(test_x)
    
    """# Add constant to normalized features
    train_x = sm.add_constant(train_x)
    test_x = sm.add_constant(test_x)"""
    
    return train_x, test_x

def print_confusion_matrix(model, test_x, test_y, predictors):
    predictions = model.predict(test_x)
    predictions_binary = [1 if p > 0.5 else 0 for p in predictions]
    cm = confusion_matrix(test_y, predictions_binary)
    print("Confusion Matrix:")
    print(cm)
    # Print the cutoff point
    cutoff_point = model.predict([[1] + [0] * (predictors.shape[1] - 1)])
    print("Cutoff Point:", cutoff_point)
    print("AIC:", model.aic)

def export_csv(train_df, test_df):
    create_csv(train_df, '/Users/juanalonso-allende/Desktop/train_dataset_normalized_with_outcome.csv')
    create_csv(test_df, '/Users/juanalonso-allende/Desktop/test_dataset_normalized_with_outcome.csv')

def main():
    df = pd.read_excel('/Users/juanalonso-allende/Desktop/Diabetes.xlsx')

    df.drop_duplicates(inplace=True)
    df.drop(["Age", "Insulin", "Skin thickness"], axis=1, inplace=True)
    print("\n\nColumns Deleted: 'Age', 'Insulin', 'Skin thickness'\n\n")
    empty_data_count = (df['Body mass index'] == 0).sum()
    print("\n\nEmpty Data in 'Body Mass' Column: ", empty_data_count, "\n\n")

    # Select the predictors and the target
    target = df["Outcome"]
    predictors = df.drop("Outcome", axis=1)

    # Normalize the features
    train_x, test_x, train_y, test_y = train_test_split(predictors, target, test_size=0.2, random_state=42)
    train_x, test_x = normalize_model(train_x, test_x)

    # Fit the model using statsmodels
    model = sm.Logit(train_y, train_x).fit()
    print("\n\nModel:\n", model.summary())
    
    # Mix the dataset
    df = df.sample(frac=1, random_state=42)
    
    print(f"\n\nTraining Data: {train_x.shape[0]} samples\nTest Data: {test_x.shape[0]} samples\n\n")
    print_confusion_matrix(model, test_x, test_y, predictors)
    
    # Create new DataFrame with normalized features
    train_df = pd.DataFrame(train_x, columns=list(predictors.columns))
    test_df = pd.DataFrame(test_x, columns=list(predictors.columns))
    train_df['Outcome'] = train_y.values
    test_df['Outcome'] = test_y.values
    
    # Call the function with the test data
    accuracy = get_model_accuracy(model, test_x, test_y)  
    print("\n\nModel Accuracy: ", accuracy, "\n\n")
    
    # Save the normalized data
    export_csv(train_df, test_df)
    
if __name__ == "__main__":
    main()