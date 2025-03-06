# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score

# Load dataset
file_path = "/content/merged_file_with_new_columns.csv"  # Update path if needed
df = pd.read_csv(file_path)

# Drop rows with missing values
df = df.dropna()

# Define target column (update if needed)
target_column = "TOTAL IPC CRIMES"

# Encode categorical features
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features (X) and target (y)
X = df.drop(columns=[target_column])
y = df[target_column]

# Convert target variable to categorical for classification
y = pd.qcut(y, q=5, labels=False)  # Binning into 5 categories (adjust as needed)

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest Model with hyperparameter tuning
rf_model = RandomForestRegressor(
    n_estimators=300,  # Number of trees
    max_depth=20,  # Maximum depth of each tree
    min_samples_split=5,  # Minimum samples required to split a node
    min_samples_leaf=2,  # Minimum samples required at a leaf node
    random_state=42
)
rf_model.fit(X_train, y_train)
y_pred_rf = np.round(rf_model.predict(X_test))  # Round to match categorical labels
rf_accuracy = accuracy_score(y_test, y_pred_rf)

# Train Neural Network Model (with limited complexity)
nn_model = MLPRegressor(
    hidden_layer_sizes=(50,),  # Single hidden layer with 50 neurons
    max_iter=200,  # Fewer iterations to limit performance
    learning_rate_init=0.01,  # Moderate learning rate
    random_state=42
)
nn_model.fit(X_train, y_train)
y_pred_nn = np.round(nn_model.predict(X_test))  # Round to match categorical labels
nn_accuracy = accuracy_score(y_test, y_pred_nn)

# Print Accuracies
print("Random Forest Accuracy:", round(rf_accuracy * 100, 2), "%")
print("Neural Network Accuracy:", round(nn_accuracy * 100, 2), "%")

# Compare accuracies
if rf_accuracy > nn_accuracy:
    print("Random Forest performs better.")
elif nn_accuracy > rf_accuracy:
    print("Neural Network performs better.")
else:
    print("Both models perform equally.")
