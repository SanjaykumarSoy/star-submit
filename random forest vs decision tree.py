import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
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
    n_estimators=300,  
    max_depth=20,  
    min_samples_split=5,  
    min_samples_leaf=2,
    random_state=42
)
rf_model.fit(X_train, y_train)
y_pred_rf = np.round(rf_model.predict(X_test))  # Round to match categorical labels
rf_accuracy = accuracy_score(y_test, y_pred_rf)

# Train Decision Tree Model
dt_model = DecisionTreeRegressor(
    max_depth=10,  
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
dt_model.fit(X_train, y_train)
y_pred_dt = np.round(dt_model.predict(X_test))  # Round to match categorical labels
dt_accuracy = accuracy_score(y_test, y_pred_dt)

# Print Accuracies
print("Random Forest Accuracy:", round(rf_accuracy * 100, 2), "%")
print("Decision Tree Accuracy:", round(dt_accuracy * 100, 2), "%")
