import 'package:flutter/material.dart';

class ChickpeaButton extends StatelessWidget {
  const ChickpeaButton({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      onPressed: () {
        // Navigate to ChickpeaPage when button is pressed
        Navigator.push(
          context,
          MaterialPageRoute(builder: (context) => ChickpeaPage()),
        );
      },
      style: ElevatedButton.styleFrom(
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(10), // Rectangular shape
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Text('Chickpea'),
      ),
    );
  }
}

class ChickpeaPage extends StatelessWidget {
  const ChickpeaPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Chickpea Details'),
      ),
      body: SingleChildScrollView(
        child: Column(
          children: <Widget>[
            Image.asset(
              'assets/chickpea.png', // Path to your chickpea image
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
                    'Variety Selection: Choose chickpea varieties suited to local climate, soil type, and intended use (e.g., desi, kabuli, or green varieties).\n'
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
                    'Timing: Plant chickpeas when soil temperatures are consistently above 50°F (10°C), typically in spring.\n'
                    'Spacing: Plant seeds in rows with appropriate spacing between plants, typically 2-4 inches (5-10 cm) apart.\n'
                    'Depth: Plant seeds at a depth of 1-2 inches (2.5-5 cm) in well-prepared soil.\n'
                    'Planting Methods: Chickpeas can be planted using mechanical planters, seed drills, or manually by hand or with dibblers.\n',
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
