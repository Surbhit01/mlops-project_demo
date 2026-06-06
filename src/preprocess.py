import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import pickle

def preprocess(data_path: str, output_dir: str):
    """
    Load and preprocess the IBM HR Attrition dataset.

    Steps:
    - Drop columns that add no information
    - Encode the target column (Attrition: Yes/No -> 1/0)
    - Encode all categorical columns using LabelEncoder
    - Split into train and test sets

    Args:
        data_path  : path to the raw CSV file
        output_dir : folder where processed files will be saved
    """

    print("=" * 50)
    print("STEP 1: Loading data")
    print("=" * 50)
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")


    # ------------------------------------------------------------------ #
    # Drop columns that carry no useful information
    # Over18     -> every employee is 'Y', zero variance
    # StandardHours -> every employee has 80, zero variance
    # EmployeeCount -> always 1
    # EmployeeNumber -> just an ID, not a feature
    # ------------------------------------------------------------------ #
    print("\nSTEP 2: Dropping low-information columns")
    cols_to_drop = ["Over18", "StandardHours", "EmployeeCount", "EmployeeNumber"]

    # Only drop columns that actually exist in the dataset
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped: {cols_to_drop}")
    print(f"Shape after dropping: {df.shape}")


    # ------------------------------------------------------------------ #
    # Encode target column
    # Attrition: Yes -> 1, No -> 0
    # ------------------------------------------------------------------ #
    print("\nSTEP 3: Encoding target column (Attrition)")
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
    print(f"Attrition value counts:\n{df['Attrition'].value_counts()}")


    # ------------------------------------------------------------------ #
    # Encode categorical columns
    # ------------------------------------------------------------------ #
    print("\nSTEP 4: Encoding categorical columns")
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    print(f"Categorical columns found: {categorical_cols}")

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    print("All categorical columns encoded.")


    # ------------------------------------------------------------------ #
    # Split features and target
    # ------------------------------------------------------------------ #
    print("\nSTEP 5: Splitting into features (X) and target (y)")
    X = df.drop(columns=["Attrition"])
    y = df["Attrition"]
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")


    # ------------------------------------------------------------------ #
    # Train / test split
    # 80% train, 20% test
    # stratify=y ensures same class ratio in both splits
    # ------------------------------------------------------------------ #
    print("\nSTEP 6: Train/test split (80/20, stratified)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape} | y_test: {y_test.shape}")


    # ------------------------------------------------------------------ #
    # Save processed data
    # ------------------------------------------------------------------ #
    print(f"\nSTEP 7: Saving processed data to '{output_dir}/'")
    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    # Save label encoders so we can reuse them during inference
    with open(f"{output_dir}/label_encoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)

    print("Saved: X_train.csv, X_test.csv, y_train.csv, y_test.csv")
    print("Saved: label_encoders.pkl")
    print("\nPreprocessing complete!")

    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------------ #
# Run directly
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    preprocess(
        data_path="data/HR-Employee-Attrition.csv",
        output_dir="data/processed"
    )