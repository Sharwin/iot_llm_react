import boto3

def get_timestream_client(profile_name='ib'):
    """
    Creates and returns an AWS Timestream query client.

    Args:
        profile_name (str): The AWS profile name to use. Defaults to 'ib'.

    Returns:
        boto3.client: A boto3 Timestream query client.
    """
    session = boto3.Session(profile_name=profile_name)
    return session.client('timestream-query')

def get_weather(variable):
    """
    Retrieves the weather variable value.

    Args:
        variable (str): The weather variable to retrieve. Valid options are:
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
        str: The latest value of the specified weather variable, or None if no data or an error occurs.
    """

    variable = variable.strip()

    client = get_timestream_client()
    query = f"""
    SELECT time, measure_value::double AS {variable}
    FROM iot.iottable
    WHERE measure_name = '{variable}'
    ORDER BY time DESC
    LIMIT 1
    """

    try:
        response = client.query(QueryString=query)
        if response['Rows']:
            return str(response['Rows'][0]['Data'][1]['ScalarValue'])
        else:
            return None
    except Exception as e:
        print(f"Error querying {variable}: {str(e)}")
        return None

def get_communication(variable):
    """
    Retrieves the latest communication variable reading from the Timestream database.

    Args:
        variable (str): The communication variable to retrieve. Valid options are:
            - rssi (Received Signal Strength Indicator)
            - snr (Signal-to-Noise Ratio)
            - freq (Frequency)
            - fCnt (Frame Count)

    Returns:
        str: The latest value of the specified communication variable, or None if no data or an error occurs.
    """
    client = get_timestream_client()
    query = f"""
    SELECT time, measure_value::double AS {variable}
    FROM iot.iottable
    WHERE measure_name = '{variable}'
    ORDER BY time DESC
    LIMIT 1
    """

    try:
        response = client.query(QueryString=query)
        if response['Rows']:
            return str(response['Rows'][0]['Data'][1]['ScalarValue'])
        else:
            return None
    except Exception as e:
        print(f"Error querying {variable}: {str(e)}")
        return None


if __name__ == "__main__":
    # Weather variables example
    temperature = get_weather("temperature")
    humidity = get_weather("humidity")
    pressure = get_weather("pressure")

    print(f"Latest temperature: {temperature}°C")
    print(f"Latest humidity: {humidity}%")
    print(f"Latest pressure: {pressure} hPa")

    # Communication variables example
    rssi = get_communication("rssi")
    snr = get_communication("snr")

    print(f"Latest RSSI: {rssi} dBm")
    print(f"Latest SNR: {snr} dB")