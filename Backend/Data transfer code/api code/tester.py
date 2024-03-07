import requests

# Define the URL of the Flask server
url = 'http://192.168.218.189:5000'

# Sample data points
data = {
    'temperature': 25.5,
    'humidity': 50.0,
    'lm35_temperature': 26.8
}

try:
    # Send GET request with data as query parameters to Flask server
    response = requests.post(url, params=data)

    # Check if the request was successful
    print(response)
    if response.status_code == 200:
        print("Data sent successfully")
    else:
        print("Failed to send data. Status code:", response.status_code)
except Exception as e:
    print("An error occurred:", e)