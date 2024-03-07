import 'package:flutter/material.dart';

class PigeonpeasButton extends StatelessWidget {
  const PigeonpeasButton({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      onPressed: () {
        // Navigate to PigeonpeasPage when button is pressed
        Navigator.push(
          context,
          MaterialPageRoute(builder: (context) => PigeonpeasPage()),
        );
      },
      style: ElevatedButton.styleFrom(
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(10), // Rectangular shape
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Text('Pigeon Peas'),
      ),
    );
  }
}

class PigeonpeasPage extends StatelessWidget {
  const PigeonpeasPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Pigeon Peas Details'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Image.asset('assets/pigeonpeas.png'), // Add image here
            SizedBox(height: 10),
            Text(
              'Land Preparation:',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            Text(
              'Site Selection: Choose a well-drained field with good soil fertility, preferably loamy or sandy loam soil.\n'
              'Clearing and Leveling: Clear the land of any weeds, rocks, and debris. Level the field to ensure uniform water distribution.\n'
              'Soil Preparation: Use plows or tillers to break up the soil and create a fine seedbed. Incorporate organic matter or compost to improve soil structure and fertility.\n',
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
              'Variety Selection: Choose pigeonpea varieties suited to local climate, soil type, and intended use (e.g., grain production, forage, or cover crop).\n'
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
