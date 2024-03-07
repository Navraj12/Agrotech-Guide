from flask import Flask, request, jsonify
import requests
import json

print("hello world")
app = Flask(__name__)
name = "Ankit"

@app.route('/api/send_data', methods=['POST'])
def receive_data():
    data = request.json
    temperature = data.get('temperature')
    humidity = data.get('humidity')
    region = data.get('region')
    print("hello")

    print("Received data - Temperature:", temperature, "Humidity:", humidity, "Region:", region)

    # Process the data as needed

    return jsonify({'message': name})

print("hello")
