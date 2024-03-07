import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

class APIpage extends StatefulWidget {
  const APIpage({Key? key}) : super(key: key);

  @override
  _APIpageState createState() => _APIpageState();
}

class _APIpageState extends State<APIpage> {
  TextEditingController latitudeController = TextEditingController();
  TextEditingController longitudeController = TextEditingController();
  dynamic outputData;
  final String accessToken = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzA5NTM2Njg3LCJpYXQiOjE3MDk1MzYzODcsImp0aSI6IjIxYzllNTFmZjMzNDQ4ODk5ZTUwZjYyOGZiZmMwNGY0IiwidXNlcl9pZCI6MzN9.oZDSgrGhvx3EIJaHiHmz6wvdLIiEv1XwQ5okkWur4_0'; // Replace with your access token

  Future<void> _submitData() async {
    final double latitude = double.tryParse(latitudeController.text) ?? 0.0;
    final double longitude = double.tryParse(longitudeController.text) ?? 0.0;

    final response = await http.get(
      Uri.parse('https://soil.narc.gov.np/soil/soildata/?lon=$longitude&lat=$latitude'),
      headers: {
        'Authorization': 'Bearer $accessToken',
      },
    );

    print('Response Body: ${response.body}');

    if (response.statusCode == 200) {
      try {
        setState(() {
          outputData = json.decode(response.body);
        });
      } catch (e) {
        print('Error decoding JSON: $e');
        setState(() {
          outputData = 'Invalid response format';
        });
      }
    } else {
      print('Failed to fetch data: ${response.statusCode}');
      setState(() {
        outputData = 'Failed to fetch data from the server';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('API Page'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Text(
              'Enter Latitude:',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            TextField(
              controller: latitudeController,
              keyboardType: TextInputType.number,
            ),
            SizedBox(height: 10),
            Text(
              'Enter Longitude:',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            TextField(
              controller: longitudeController,
              keyboardType: TextInputType.number,
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: _submitData,
              child: Text('Submit'),
            ),
            SizedBox(height: 20),
            Text(
              'Output:',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            SizedBox(height: 10),
            _buildOutputWidget(),
          ],
        ),
      ),
    );
  }

  Widget _buildOutputWidget() {
  if (outputData != null) {
    if (outputData is Map<String, dynamic>) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: outputData.entries.map((entry) {
          return Text('${entry.key}: ${entry.value}');
        }).toList(),
      );
    } else if (outputData is List<dynamic>) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: outputData.map((item) {
          return Text(item.toString());
        }).toList(),
      );
    } else if (outputData is String) {
      return Text(outputData);
    } else {
      // Handle unexpected data type
      return Text('Unexpected output data type: ${outputData.runtimeType}');
    }
  } else {
    return Container();
  }
}
}

