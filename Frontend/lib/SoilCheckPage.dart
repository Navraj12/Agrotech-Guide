import 'dart:convert';
import 'dart:ffi';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

class SoilCheckPage extends StatefulWidget {
  @override
  _SoilCheckPageState createState() => _SoilCheckPageState();
}

class _SoilCheckPageState extends State<SoilCheckPage> {
  final TextEditingController nController = TextEditingController();
  final TextEditingController pController = TextEditingController();
  final TextEditingController kController = TextEditingController();
  final TextEditingController temperatureController = TextEditingController();
  final TextEditingController humidityController = TextEditingController();
  final TextEditingController phController = TextEditingController();
  final TextEditingController rainfallController = TextEditingController();
  String soilStatus = '';
  String cropImage = '';

  // Mapping of soil status to corresponding crop image file names
  final Map<String, String> cropImageMap = {
    'rice': 'rice.png',
    'maize': 'maize.png',
    'chickpea': 'chickpea.png',
    'kidneybeans': 'kidneybeans.png',
    'pigeonpeas': 'pigeonpeas.png',
    'mothbeans': 'mothbeans.png',
    'mungbean': 'mungbean.png',
    'blackgram': 'blackgram.png',
    'lentil': 'lentil.png',
    'pomegranate': 'pomegranate.png',
    'banana': 'banana.png',
    'mango': 'mango.png',
    'grapes': 'grapes.png',
    'watermelon': 'watermelon.png',
    'muskmelon': 'muskmelon.png',
    'apple': 'apple.png',
    'orange': 'orange.png',
    'papaya': 'papaya.png',
    'coconut': 'coconut.png',
    'cotton': 'cotton.png',
    'jute': 'jute.png',
    'coffee': 'coffee.png',
  };
  
  //get temperatureValue => null;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Check Your Soil'),
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: <Widget>[
              buildTextField('N', nController),
              buildTextField('P', pController),
              buildTextField('K', kController),
              //buildTextField('Temperature', temperatureController),              
              //buildTextField('Humidity', humidityController),
              buildTextField('pH', phController),
              buildTextField('Rainfall', rainfallController),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: () {
                  _checkSoil();
                },
                child: Text(
                  'Check Soil',
                  style: TextStyle(color: Colors.white), // Change text color to red
                ),
                style: ElevatedButton.styleFrom(
                  primary: Color.fromARGB(255, 70, 156, 106),
                ),
              ),
              const SizedBox(height: 20),
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: <Widget>[
                  Text(
                    'Soil Status: $soilStatus',
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 20),
                  if (cropImage.isNotEmpty)
                    Image.asset('assets/$cropImage')
                  else
                    SizedBox(), // Show image if cropImage is not empty
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget buildTextField(String label, TextEditingController controller) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Enter $label',
          style: TextStyle(fontSize: 16),
        ),
        TextField(
          controller: controller,
          keyboardType: TextInputType.number,
        ),
        const SizedBox(height: 20),
      ],
    );
  }

  Future<void> _checkSoil() async {
    final nValue = nController.text;
    final pValue = pController.text;
    final kValue = kController.text;
    if (nValue == 35) {
      double temperatureValue = 28;
      
    } else {
    double temperatureValue = 20;
             }
    final temperatureValue = temperatureController.text;

    //final humidityValue = 81;

    if (pValue == 41) {
      double humidityValue = 81;
      
    } else {
      double humidityValue = 81;
    }
    //final temperatureValue = temperatureController.text;
    final humidityValue = humidityController.text;
    final phValue = phController.text;
    final rainfallValue = rainfallController.text;

    final url = Uri.parse('http://192.168.218.64:5000/api/send_data');

    try {
      final response = await http.post(
        url,
        body: json.encode({
          'N': nValue,
          'P': pValue,
          'K': kValue,
          'temperature': temperatureValue,
          'humidity': humidityValue,
          'pH': phValue,
          'rainfall': rainfallValue,
        }),
        headers: {'Content-Type': 'application/json'},
      );

      if (response.statusCode == 200) {
        setState(() {
          soilStatus = response.body;
          cropImage = cropImageMap[soilStatus.toLowerCase()] ?? ''; // Get corresponding image filename
        });
      } else {
        setState(() {
          soilStatus = 'Failed to check soil status. Error code: ${response.statusCode}';
          cropImage = '';
        });
      }
    } catch (e) {
      setState(() {
        soilStatus = 'Failed to check soil status. Error: $e';
        cropImage = '';
      });
    }
  }
}
