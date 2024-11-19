from cgitb import reset
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai.chat_models import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
# from langchain_fireworks import ChatFireworks
import boto3
import joblib
import io
from datetime import datetime, timezone, timedelta
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
import requests
from botocore.exceptions import ClientError
import os
import json

load_dotenv()

react_prompt: PromptTemplate = hub.pull("hwchase17/react")


def get_timestream_client(profile_name="ib"):
    """
    Creates and returns an AWS Timestream query client.

    Args:
        profile_name (str, optional): The AWS profile name to use. Defaults to 'ib'.

    Returns:
        boto3.client: A boto3 Timestream query client.

    Example:
        >>> client = get_timestream_client()
    """
    session = boto3.Session(profile_name=profile_name)
    return session.client("timestream-query")

def get_s3_client(profile_name="ib"):
    """
    Creates and returns an AWS S3 client.

    Args:
        profile_name (str): The AWS profile name to use. Defaults to 'ib'.

    Returns:
        boto3.client: A boto3 S3 client.

    Example:
        >>> s3 = get_s3_client()
    """
    session = boto3.Session(profile_name=profile_name)
    return session.client("s3")

@tool
def get_weather(weather_variable: str) -> str:
    """
    Retrieves the latest value of the specified weather variable from Timestream.

    Args:
        weather_variable (str): The weather variable to retrieve.
            Valid options include:
                - temperature
                - light
                - cloud_base_altitude
                - dew_point
                - actual_vapor_pressure
                - heat_index
                - saturation_vapor_pressure
                - pressure
                - humidity
                - relative_humidity

    Returns:
        str: The latest value of the specified weather variable,
             formatted with appropriate units, or an error message if retrieval fails.

    Example:
        >>> get_weather("temperature")
        "The latest temperature reading is 22.5°C."
    """
    weather_variable = weather_variable.strip()
    client = get_timestream_client()
    query = f"""
    SELECT time, measure_value::double AS {weather_variable}
    FROM iot.iottable
    WHERE measure_name = '{weather_variable}'
    ORDER BY time DESC
    LIMIT 1
    """

    try:
        response = client.query(QueryString=query)
        if response.get('Rows'):
            value = response['Rows'][0]['Data'][1]['ScalarValue']
            unit = {
                "temperature": "°C",
                "humidity": "%",
                "pressure": "hPa",
                "light": "lumens"
            }.get(weather_variable, "")
            return f"The latest {weather_variable} reading is {value}{unit}."
        else:
            return "No data found for the specified weather variable."
    except Exception as e:
        return f"Error querying {weather_variable}: {str(e)}"


@tool
def get_communication(communication_variable: str) -> str:
    """
    Retrieves the latest value of the specified communication variable from Timestream.

    Args:
        communication_variable (str): The communication variable to retrieve.
            Valid options include:
                - rssi (Received Signal Strength Indicator)
                - snr (Signal-to-Noise Ratio)
                - freq (Frequency)
                - fCnt (Frame Count)

    Returns:
        str: The latest value of the specified communication variable,
             formatted with appropriate units, or an error message if retrieval fails.

    Example:
        get_communication("rssi")
        "The latest RSSI value is -70 dBm."
    """
    communication_variable = communication_variable.strip()
    client = get_timestream_client()
    query = f"""
    SELECT time, measure_value::double AS {communication_variable}
    FROM iot.iottable
    WHERE measure_name = '{communication_variable}'
    ORDER BY time DESC
    LIMIT 1
    """

    try:
        response = client.query(QueryString=query)
        if response.get('Rows'):
            value = response['Rows'][0]['Data'][1]['ScalarValue']
            unit = {
                "rssi": "dBm",
                "snr": "dB",
                "freq": "MHz",
                "fCnt": ""
            }.get(communication_variable, "")
            return f"The latest {communication_variable.upper()} value is {value}{unit}."
        else:
            return "No data found for the specified communication variable."
    except Exception as e:
        return f"Error querying {communication_variable}: {str(e)}"


@tool
def get_current_time(input: str = "") -> str:
    """
    Retrieves the current UTC time.

    Args:
        input (str, optional): An optional input string. This tool does not utilize the input. Defaults to "".

    Returns:
        str: Current UTC time in ISO 8601 format.

    Example:
        >>> get_current_time()
        "Current UTC time is 2024-04-27T14:23:45Z."
    """
    current_time = datetime.now(timezone.utc).isoformat()
    return f"Current UTC time is {current_time}."


@tool
def load_model_and_predict(feature_str: str) -> str:
    """
    Loads the trained model from S3 and makes a prediction based on input features.

    Args:
        feature_str (str): Comma-separated feature values required for prediction.
            The expected order of features is:
                1. Hour (0-23)
                2. Day of Year (1-366)
                3. Humidity (%)
                4. Pressure (hPa)
                5. Light (lumens)

    Returns:
        str: Prediction result formatted with appropriate units, or an error message if prediction fails.

    Example:
        load_model_and_predict("14, 100, 75, 1013, 300")
        "Predicted temperature for the next hour is 23.8°C."
    """
    try:
        features = [float(x.strip()) for x in feature_str.split(',')]
        if len(features) != 5:
            return "Invalid number of features. Expected 5 numerical values separated by commas."
    except ValueError:
        return "Invalid feature input. Please provide comma-separated numerical values."

    BUCKET_NAME = os.getenv('MODEL_BUCKET_NAME', 'training-models-ivan')
    MODEL_OBJECT_KEY = os.getenv('MODEL_OBJECT_KEY', 'models/temperature_model.joblib')

    s3_client = get_s3_client()
    try:
        buffer = io.BytesIO()
        s3_client.download_fileobj(Bucket=BUCKET_NAME, Key=MODEL_OBJECT_KEY, Fileobj=buffer)
        buffer.seek(0)
        model = joblib.load(buffer)
    except Exception as e:
        return f"Failed to load model from S3: {str(e)}"

    try:
        prediction = model.predict([features])[0]
        return f"Predicted temperature for the next hour is {prediction:.2f}°C."
    except Exception as e:
        return f"Prediction failed: {str(e)}"


@tool
def evaluate_model_performance(input_str: str) -> str:
    """
    Evaluates the model's performance using Mean Squared Error (MSE), Root Mean Squared Error (RMSE),
    and R² Score metrics, and makes a temperature prediction for the next hour.

    This function parses the input string containing feature values and the actual target value,
    loads a pre-trained model from an S3 bucket, makes a prediction, and returns the evaluation metrics
    and prediction in a structured JSON format.

    Args:
        input_str (str): Comma-separated values containing feature values followed by the actual target value.
            The expected order of values is:
                1. Hour (0-23)
                2. Day of Year (1-366)
                3. Humidity (%)
                4. Pressure (hPa)
                5. Light (lumens)
                6. Actual Temperature Value (float)

            Example:
                "14, 100, 75, 1013, 300, 23.5"

    Returns:
        str: A JSON string containing evaluation metrics and the predicted temperature,
             or an error message if evaluation fails.

    Example:
        >>> evaluate_model_performance("14, 100, 75, 1013, 300, 23.5")
        '{"evaluation_metrics": {"MSE": 0.09, "RMSE": 0.3, "R2_Score": 0.95}, "predicted_temperature": 25.7}'
    """
    # Parse the input values
    try:
        values = [float(x.strip()) for x in input_str.split(',')]
        if len(values) != 6:
            raise ValueError(f"Invalid number of values. Expected 6 (5 features + 1 actual value), got {len(values)}.")

        features = values[:5]
        actual = values[5]
    except ValueError as ve:
        error_response = {
            "error": "Input Parsing Error",
            "message": str(ve)
        }
        return json.dumps(error_response)

    hour, day_of_year, humidity, pressure, light = features
    if not (0 <= hour <= 23):
        error_response = {
            "error": "Validation Error",
            "message": f"Hour value {hour} is out of range (0-23)."
        }
        return json.dumps(error_response)
    if not (1 <= day_of_year <= 366):
        error_response = {
            "error": "Validation Error",
            "message": f"Day of Year value {day_of_year} is out of range (1-366)."
        }
        return json.dumps(error_response)
    if not (0 <= humidity <= 100):
        error_response = {
            "error": "Validation Error",
            "message": f"Humidity value {humidity} is out of range (0-100%)."
        }
        return json.dumps(error_response)
    if not (10 <= pressure <= 1200):
        error_response = {
            "error": "Validation Error",
            "message": f"Pressure value {pressure} is out of typical range (10-1200 hPa)."
        }
        return json.dumps(error_response)
    if not (0 <= light <= 100000):
        error_response = {
            "error": "Validation Error",
            "message": f"Light value {light} is out of range (0-100000 lumens)."
        }
        return json.dumps(error_response)

    # Define S3 parameters
    BUCKET_NAME = os.getenv('MODEL_BUCKET_NAME', 'training-models-ivan')
    MODEL_OBJECT_KEY = os.getenv('MODEL_OBJECT_KEY', 'models/temperature_model.joblib')

    s3_client = get_s3_client()
    try:
        buffer = io.BytesIO()
        s3_client.download_fileobj(Bucket=BUCKET_NAME, Key=MODEL_OBJECT_KEY, Fileobj=buffer)
        buffer.seek(0)
        model = joblib.load(buffer)
    except Exception as e:
        error_response = {
            "error": "Model Loading Error",
            "message": f"Failed to load model from S3. Details: {str(e)}"
        }
        return json.dumps(error_response)

    try:
        prediction = model.predict([features])[0]
        mse = mean_squared_error([actual], [prediction])
        rmse = np.sqrt(mse)
        r2 = r2_score([actual], [prediction])

        r2_value = r2 if not np.isnan(r2) else "Not available (likely due to insufficient data variability)"

        response = {
            "evaluation_metrics": {
                "MSE": round(mse, 4),
                "RMSE": round(rmse, 4),
                "R2_Score": r2_value
            },
            "predicted_temperature": round(prediction, 2)
        }

        return json.dumps(response)
    except Exception as e:
        error_response = {
            "error": "Evaluation Error",
            "message": f"Failed to compute metrics. Details: {str(e)}"
        }
        return json.dumps(error_response)

@tool
def detect_anomalies_from_iot_sensor(sensor_variable: str) -> str:
    """
    Retrieves values from AWS Timestream for the specified sensor and determines if there are anomalous values.

    Args:
        sensor_variable (str): The IoT sensor variable to analyze for anomalies.

    Returns:
        str: A summary of detected anomalies or a message indicating no anomalies were found.

    Example:
        >>> detect_anomalies_from_iot_sensor("temperature")
        "Anomalies detected: 2 out of 100 data points."
    """
    sensor_variable = sensor_variable.strip()
    client = get_timestream_client()

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=24)
    time_range_start = start_time.isoformat()
    time_range_end = end_time.isoformat()

    query = f"""
    SELECT time, measure_value::double AS {sensor_variable}
    FROM iot.iottable
    WHERE measure_name = '{sensor_variable}'
      AND time BETWEEN from_iso8601_timestamp('{time_range_start}') AND from_iso8601_timestamp('{time_range_end}')
    ORDER BY time ASC
    """

    try:
        response = client.query(QueryString=query)
        records = response.get('Rows', [])
        if not records:
            return f"No data found for sensor variable '{sensor_variable}'."

        timestamps = [row['Data'][0]['ScalarValue'] for row in records]
        values = [float(row['Data'][1]['ScalarValue']) for row in records]
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps),
            sensor_variable: values
        })

        model = IsolationForest(contamination=0.05)  # 5% contamination
        df['anomaly'] = model.fit_predict(df[[sensor_variable]])
        anomalies = df[df['anomaly'] == -1]

        if not anomalies.empty:
            anomaly_count = len(anomalies)
            total_count = len(df)
            return f"Anomalies detected: {anomaly_count} out of {total_count} data points."
        else:
            return "No anomalies detected in the sensor data."

    except ClientError as e:
        return f"Error querying {sensor_variable}: {e}"
    except Exception as e:
        return f"An error occurred during anomaly detection: {str(e)}"

@tool
def get_weather_openweathermap(weather_variable: str) -> str:
    """
    Retrieves the latest value of the specified weather variable using the OpenWeatherMap API.

    Args:
        weather_variable (str): The weather variable to retrieve.
            Valid options include:
                - temperature
                - humidity
                - pressure
                - wind_speed
                - wind_deg
                - clouds
                - visibility

    Returns:
        str: The latest value of the specified weather variable,
             formatted with appropriate units, or an error message if retrieval fails.

    Example:
        >>> get_weather_openweathermap("temperature")
        "The current temperature is 15.96°C."
    """
    weather_variable = weather_variable.strip().lower()
    api_key = os.getenv('OPENWEATHER_API_KEY', '')
    if not api_key:
        return "OpenWeatherMap API key is not configured."

    lat = 4.653529
    lon = -74.074147

    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"

    try:
        response = requests.get(url)
        data = response.json()

        if response.status_code != 200:
            error_message = data.get('message', 'Unknown error.')
            return f"Failed to get data. Error: {error_message}"

        # Mapping of weather variables to their paths in the API response
        variable_mapping = {
            "temperature": data['main']['temp'],
            "humidity": data['main']['humidity'],
            "pressure": data['main']['pressure'],
            "wind_speed": data['wind']['speed'],
            "wind_deg": data['wind']['deg'],
            "clouds": data['clouds']['all'],
            "visibility": data['visibility']
        }

        if weather_variable not in variable_mapping:
            return f"Invalid weather variable '{weather_variable}'. Valid options are: {', '.join(variable_mapping.keys())}."

        value = variable_mapping[weather_variable]
        unit = {
            "temperature": "°C",
            "humidity": "%",
            "pressure": "hPa",
            "wind_speed": "m/s",
            "wind_deg": "°",
            "clouds": "%",
            "visibility": "meters"
        }.get(weather_variable, "")

        if weather_variable == "light":
            return f"The current weather condition is '{value}'."
        else:
            return f"The current {weather_variable.replace('_', ' ')} is {value}{unit}."

    except Exception as e:
        return f"Error retrieving weather data: {str(e)}"

tools = [
    get_weather,
    get_communication,
    get_current_time,
    load_model_and_predict,
    evaluate_model_performance,
    detect_anomalies_from_iot_sensor,
    get_weather_openweathermap
]

llm = ChatOpenAI(model="gpt-4o")
# llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
# llm = ChatFireworks(model="accounts/fireworks/models/yi-large")
# llm = ChatFireworks(model="accounts/fireworks/models/llama-v3-70b-instruct")

react_agent_runnable = create_react_agent(llm, tools, react_prompt)


