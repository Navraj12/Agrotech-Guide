from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api')
def hello_world():
    return jsonify({"message": "hello world"})

@app.route('/', methods=['GET'])
def receive_data():
    if request.method == 'GET':
        temperature = request.args.get('temperature')  # Extract temperature from query parameters
        humidity = request.args.get('humidity')  # Extract humidity from query parameters
        lm35_temperature = request.args.get('lm35_temperature')  # Extract LM35 temperature from query parameters
        
        print("Temperature:", temperature)
        print("Humidity:", humidity)
        print("LM35 Temperature:", lm35_temperature)
        
        # Process sensor data as needed
        
        response_data = {
            'temperature': temperature,
            'humidity': humidity,
            'lm35_temperature': lm35_temperature
        }
        
        return jsonify(response_data), 200  # Send JSON response to the client

if __name__ == '__main__':
    app.run(host='192.168.218.64', port=5000)  # Run the Flask app on port 5000