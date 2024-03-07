import 'package:flutter/material.dart';

class MungbeanButton extends StatelessWidget {
  const MungbeanButton({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      onPressed: () {
        // Navigate to MungbeanPage when button is pressed
        Navigator.push(
          context,
          MaterialPageRoute(builder: (context) => MungbeanPage()),
        );
      },
      style: ElevatedButton.styleFrom(
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(10), // Rectangular shape
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Text('Mung Bean'),
      ),
    );
  }
}

class MungbeanPage extends StatelessWidget {
  const MungbeanPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Mung Bean Details'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Image.asset(
              'assets/mungbean.png', // Path to your mung bean image
              fit: BoxFit.cover,
            ),
            SizedBox(height: 10),
            Text(
              'Land Preparation:',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            Text(
              'Site Selection: Choose a well-drained field with good soil fertility and adequate sunlight.\n'
              'Clearing and Leveling: Clear the land of any existing vegetation, rocks, and debris. Level the field to ensure uniform water distribution.\n'
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
              'Variety Selection: Choose mung bean varieties suited to local climate, soil type, and intended use.\n'
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
