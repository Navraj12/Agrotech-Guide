import 'package:flutter/material.dart';

class MothbeansButton extends StatelessWidget {
  const MothbeansButton({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      onPressed: () {
        // Navigate to MothbeansPage when button is pressed
        Navigator.push(
          context,
          MaterialPageRoute(builder: (context) => MothbeansPage()),
        );
      },
      style: ElevatedButton.styleFrom(
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(10), // Rectangular shape
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Text('Mothbeans'),
      ),
    );
  }
}

class MothbeansPage extends StatelessWidget {
  const MothbeansPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Mothbeans Details'),
      ),
      body: SingleChildScrollView(
        child: Column(
          children: <Widget>[
            // Add image here
            Image.asset(
              'assets/mothbeans.png', // Path to your mothbeans image
              fit: BoxFit.cover,
            ),
            Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: <Widget>[
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
                    'Variety Selection: Choose moth bean varieties suited to local climate, soil type, and intended use.\n'
                    'Seed Treatment: Treat seeds with fungicides and insecticides to protect against soil-borne diseases and pests.\n'
                    'Seed Testing: Conduct germination tests to ensure seed viability and vigor.\n',
                    style: TextStyle(fontSize: 16),
                  ),
                  SizedBox(height: 10),
                  Text(
                    'Planting:',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Text(
                    'Timing: Plant moth beans when soil temperatures are warm, typically after the last frost in spring.\n'
                    'Spacing: Plant seeds in rows with appropriate spacing between plants, typically 8-12 inches (20-30 cm) apart.\n'
                    'Depth: Plant seeds at a depth of 1-2 inches (2.5-5 cm) in well-prepared soil.\n'
                    'Planting Methods: Moth beans can be planted using mechanical planters, seed drills, or manually by hand or with dibblers.\n',
                    style: TextStyle(fontSize: 16),
                  ),
                  // Add other sections similarly...
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
