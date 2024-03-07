from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api')
def hello_world():
    return 'Hello, World!'

@app.route('/', methods=['POST'])
def receive_data():
    data = request.json
    print("hello", len(data))
    print("Received Data from Arduino:", data)

@app.route('/api1', methods=['POST'])
def receive_data1():
    if request.method == 'POST':
        print("Entire Request:")
        print(request.dict)
        temperature = request.args.get('temperature')
        humidity = request.args.get('humidity')
        lm35_temperature = request.args.get('lm35_temperature')

        print("Temperature:", temperature)
        print("Humidity:", humidity)
        print("LM35 Temperature:", lm35_temperature)
        
        response_data = {
            'temperature': temperature,
            'humidity': humidity,
            'lm35_temperature': lm35_temperature
        }
        
        return jsonify(response_data), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

    #app.run(host='192.168.218.64', port=5000, debug=True)


