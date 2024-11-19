import boto3
from botocore.exceptions import ClientError
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import io
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json



def get_timestream_client(region_name='us-east-1', profile_name=None):
    """
    Initializes and returns a Timestream query client.

    Args:
        region_name (str): AWS region name.
        profile_name (str, optional): AWS CLI profile name. Defaults to None.

    Returns:
        boto3.client: Timestream query client.
    """
    if profile_name:
        session = boto3.Session(profile_name=profile_name)
        return session.client('timestream-query', region_name=region_name)
    else:
        return boto3.client('timestream-query', region_name=region_name)

def get_s3_client(region_name='us-east-1', profile_name=None):
    """
    Initializes and returns an S3 client.

    Args:
        region_name (str): AWS region name.
        profile_name (str, optional): AWS CLI profile name. Defaults to None.

    Returns:
        boto3.client: S3 client.
    """
    if profile_name:
        session = boto3.Session(profile_name=profile_name)
        return session.client('s3', region_name=region_name)
    else:
        return boto3.client('s3', region_name=region_name)


def query_timestream(client, measure_names, database, table, time_range_start, time_range_end):
    """
    Queries Timestream for the specified measure names within the given time range.

    Args:
        client (boto3.client): Timestream query client.
        measure_names (list): List of measure names to retrieve.
        database (str): Timestream database name.
        table (str): Timestream table name.
        time_range_start (str): Start time in ISO 8601 format.
        time_range_end (str): End time in ISO 8601 format.

    Returns:
        pd.DataFrame: Combined DataFrame with all requested measures.
    """
    data_frames = []

    for measure in measure_names:
        query = f"""
            SELECT time, measure_value::double AS {measure}
            FROM "{database}"."{table}"
            WHERE measure_name = '{measure}'
              AND time BETWEEN from_iso8601_timestamp('{time_range_start}') AND from_iso8601_timestamp('{time_range_end}')
            ORDER BY time ASC
        """
        try:
            response = client.query(QueryString=query)
            records = response.get('Rows', [])
            timestamps = [row['Data'][0]['ScalarValue'] for row in records]
            values = [float(row['Data'][1]['ScalarValue']) for row in records]
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(timestamps),
                measure: values
            })
            data_frames.append(df)
            print(f"Retrieved {len(df)} records for measure '{measure}'.")
        except ClientError as e:
            print(f"Error querying {measure}: {e}")
            continue

    if data_frames:
        combined_df = data_frames[0]
        for df in data_frames[1:]:
            combined_df = pd.merge(combined_df, df, on='timestamp', how='inner')
        print(f"Combined DataFrame has {len(combined_df)} records.")
        return combined_df
    else:
        print("No data retrieved from Timestream.")
        return pd.DataFrame()



def preprocess_data(df):
    """
    Preprocesses the DataFrame by adding time-based features and handling missing values.

    Args:
        df (pd.DataFrame): Raw DataFrame with sensor data.

    Returns:
        pd.DataFrame: Preprocessed DataFrame ready for training.
    """
    if df.empty:
        raise ValueError("The DataFrame is empty. Cannot preprocess data.")

    # Feature Engineering: Extract hour and day from timestamp
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.dayofyear

    # Handle missing values if any
    initial_length = len(df)
    df = df.dropna()
    final_length = len(df)
    if initial_length != final_length:
        print(f"Dropped {initial_length - final_length} rows due to missing values.")

    return df



def train_temperature_model(df):
    """
    Trains a Random Forest Regressor to predict temperature.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame with features and target.

    Returns:
        RandomForestRegressor: Trained Random Forest model.
        float: Mean Squared Error on the test set.
    """
    features = ['hour', 'day', 'humidity', 'pressure', 'light']
    target = 'temperature'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training completed.")

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Temperature Model Mean Squared Error (MSE): {mse}")

    return model, mse, X_test, y_test, predictions


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model using various metrics.

    Args:
        model: Trained machine learning model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Actual target values.

    Returns:
        tuple: MSE, RMSE, R² Score, and predictions.
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)
    print(f"Model Evaluation Metrics:")
    print(f" - MSE: {mse}")
    print(f" - RMSE: {rmse}")
    print(f" - R² Score: {r2}")
    return mse, rmse, r2, predictions


def tune_hyperparameters(X_train, y_train):
    """
    Tunes hyperparameters of the Random Forest model using GridSearchCV.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target values.

    Returns:
        RandomForestRegressor: Best estimator found by GridSearchCV.
    """
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        verbose=2
    )

    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best MSE: {-grid_search.best_score_}")

    return grid_search.best_estimator_


def save_model_to_s3(model, bucket_name, object_key, s3_client):
    """
    Saves the trained model to an S3 bucket.

    Args:
        model: Trained machine learning model.
        bucket_name (str): Name of the S3 bucket.
        object_key (str): S3 object key (path) where the model will be saved.
        s3_client (boto3.client): Initialized S3 client.

    Returns:
        bool: True if upload is successful, False otherwise.
    """
    try:
        # Serialize the model to a bytes buffer
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)

        # Upload the model to S3
        s3_client.upload_fileobj(buffer, Bucket=bucket_name, Key=object_key)
        print(f"Model successfully saved to s3://{bucket_name}/{object_key}")
        return True
    except ClientError as e:
        print(f"Failed to upload model to S3: {e}")
        return False


def load_model_from_s3(bucket_name, object_key, s3_client):
    """
    Loads a trained model from an S3 bucket.

    Args:
        bucket_name (str): Name of the S3 bucket.
        object_key (str): S3 object key (path) where the model is saved.
        s3_client (boto3.client): Initialized S3 client.

    Returns:
        model: Loaded machine learning model, or None if failed.
    """
    try:
        buffer = io.BytesIO()
        s3_client.download_fileobj(Bucket=bucket_name, Key=object_key, Fileobj=buffer)
        buffer.seek(0)
        model = joblib.load(buffer)
        print("Model successfully loaded from S3.")
        return model
    except ClientError as e:
        print(f"Failed to load model from S3: {e}")
        return None


def plot_residuals(y_true, y_pred):
    """
    Plots residuals to evaluate model assumptions.

    Args:
        y_true (pd.Series): Actual target values.
        y_pred (np.array): Predicted target values.
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values')
    plt.show()

def plot_feature_importances(model, feature_names):
    """
    Plots feature importances from the trained model.

    Args:
        model: Trained machine learning model.
        feature_names (list): List of feature names.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10,6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()


def save_performance_metrics(mse, rmse, r2, filepath='model_performance.json'):
    """
    Saves performance metrics to a JSON file.

    Args:
        mse (float): Mean Squared Error.
        rmse (float): Root Mean Squared Error.
        r2 (float): R² Score.
        filepath (str): Path to save the JSON file.
    """
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'R2_Score': r2
    }
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {filepath}")


def cross_validate_model(model, X, y, cv=5):
    """
    Performs cross-validation on the model.

    Args:
        model: Machine learning model.
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
        cv (int, optional): Number of cross-validation folds. Defaults to 5.
    """
    mse_scores = -cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)
    print(f"Cross-Validated MSE scores: {mse_scores}")
    print(f"Average MSE: {np.mean(mse_scores)}")
    print(f"Standard Deviation of MSE: {np.std(mse_scores)}")


def main():
    REGION = 'us-east-1'
    DATABASE = 'iot' 
    TABLE = 'table'
    BUCKET_NAME = 'name'  # Replace with your S3 bucket name
    MODEL_OBJECT_KEY = 'models/temperature_model.joblib'  # Path in S3 bucket to save the model
    PROFILE_NAME = 'default'  # Replace with your AWS CLI profile name or set to None

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=7)
    time_range_start = start_time.isoformat()
    time_range_end = end_time.isoformat()

    measures = ['temperature', 'light', 'humidity', 'pressure']

    print("Initializing AWS clients...")
    timestream_client = get_timestream_client(region_name=REGION, profile_name=PROFILE_NAME)
    s3_client = get_s3_client(region_name=REGION, profile_name=PROFILE_NAME)
    print("AWS clients initialized.")

    print("Querying Timestream for historical data...")
    df = query_timestream(
        client=timestream_client,
        measure_names=measures,
        database=DATABASE,
        table=TABLE,
        time_range_start=time_range_start,
        time_range_end=time_range_end
    )

    if df.empty:
        print("No data retrieved from Timestream. Exiting.")
        return

    print("Preprocessing data...")
    df_preprocessed = preprocess_data(df)

    print("Training the Random Forest model...")
    model, mse, X_test, y_test, predictions = train_temperature_model(df_preprocessed)

    print("Evaluating the model...")
    mse, rmse, r2, predictions = evaluate_model(model, X_test, y_test)

    save_performance_metrics(mse, rmse, r2)
    plot_feature_importances(model, X_test.columns.tolist())
    plot_residuals(y_test, predictions)

    print("Performing cross-validation...")
    cross_validate_model(model, df_preprocessed[['hour', 'day', 'humidity', 'pressure', 'light']], df_preprocessed['temperature'])

    print("Saving the trained model to S3...")
    success = save_model_to_s3(model, BUCKET_NAME, MODEL_OBJECT_KEY, s3_client)

    if success:
        print("Model training and saving completed successfully.")
    else:
        print("Model training completed, but saving to S3 failed.")


if __name__ == "__main__":
    main()
