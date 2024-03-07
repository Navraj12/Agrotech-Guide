import 'package:flutter/material.dart';

class KidneybeansButton extends StatelessWidget {
  const KidneybeansButton({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      onPressed: () {
        // Navigate to KidneybeansPage when button is pressed
        Navigator.push(
          context,
          MaterialPageRoute(builder: (context) => KidneybeansPage()),
        );
      },
      style: ElevatedButton.styleFrom(
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(10), // Rectangular shape
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Text('Kidney Beans'),
      ),
    );
  }
}

class KidneybeansPage extends StatelessWidget {
  const KidneybeansPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Kidney Beans Details'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Image.asset('assets/kidneybeans.png'), // Add image here
            SizedBox(height: 10),
            Text(
              'Land Preparation:',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            Text(
              'Site Selection: Choose a well-drained field with good soil fertility, adequate sunlight, and proper air circulation.\n'
              'Clearing: Clear the land of any existing vegetation, rocks, and debris.\n'
              'Soil Testing: Conduct soil tests to determine nutrient levels, pH, and soil composition.\n'
              'Soil Preparation: Use plows or tillers to break up the soil and create a suitable seedbed. Incorporate organic matter or compost to improve soil structure and fertility.\n',
              style: TextStyle(fontSize: 16),
            ),
            SizedBox(height: 10),
            Text(
              'Seed Selection and Preparation:',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            Text(
              'Variety Selection: Choose kidney bean varieties suited to local climate, soil type, and intended use (e.g., bush or pole beans, dry or snap beans).\n'
              'Seed Treatment: Treat seeds with fungicides and insecticides to protect against soil-borne diseases and pests.\n'
              'Seed Testing: Conduct germination tests to ensure seed viability and vigor.\n',
              style: TextStyle(fontSize: 16),
            ),
            // Add other sections similarly...
          ],
        ),
      ),
    );
  }
}
