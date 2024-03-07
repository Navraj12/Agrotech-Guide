import 'package:flutter/material.dart';

class BananaButton extends StatelessWidget {
  const BananaButton({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      onPressed: () {
        Navigator.push(
          context,
          MaterialPageRoute(builder: (context) => RicePage()),
        );
      },
      style: ElevatedButton.styleFrom(
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(10), // Rectangular shape
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Text('Banana'),
      ),
    );
  }
}

class RicePage extends StatelessWidget {
  const RicePage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Banana Details'),
      ),
      body: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: Image.asset(
                'assets/banana.png', // Path to the rice image
                height: 200,
                width: 400, // Adjust the height as needed
              ),
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
                    'Selection of Land: Choose a suitable plot of land with access to water for flooding or irrigation.\n'
                    'Clearing and Leveling: Clear the land of any vegetation and debris. Level the field to ensure uniform flooding and water distribution.\n'
                    'Plowing and Tilling: Use plows or tillers to break up the soil and prepare the seedbed. This helps improve soil aeration and water infiltration.\n',
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
                    'Variety Selection: Choose appropriate rice varieties based on factors such as climate, soil type, and market demand.\n'
                    'Seed Treatment: Treat seeds with fungicides or insecticides to protect against diseases and pests.\n'
                    'Seed Soaking: Soak seeds in water for a specific period to encourage germination and uniform sprouting.\n',
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
                    'Direct Seeding: Broadcast seeds directly into the flooded field or use seed drills for precise spacing.\n'
                    'Transplanting: Alternatively, transplant seedlings from nurseries into the flooded field. This method requires growing seedlings in a separate nursery field before transplanting.\n',
                    style: TextStyle(fontSize: 16),
                  ),
                  SizedBox(height: 10),
                  Text(
                    'Water Management:',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Text(
                    'Flooding: Maintain flooded conditions in the field throughout most of the growing season. This suppresses weed growth and provides a suitable environment for rice plants.\n'
                    'Intermittent Flooding: In some systems, water is drained periodically to allow oxygen to reach the roots, preventing waterlogging.\n',
                    style: TextStyle(fontSize: 16),
                  ),
                  SizedBox(height: 10),
                  Text(
                    'Fertilization and Weed Control:',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Text(
                    'Fertilization: Apply fertilizers, both organic and synthetic, to provide essential nutrients like nitrogen, phosphorus, and potassium.\n'
                    'Weed Control: Use herbicides, manual weeding, or mechanical methods to control weeds that compete with rice plants for nutrients and space.\n',
                    style: TextStyle(fontSize: 16),
                  ),
                  SizedBox(height: 10),
                  Text(
                    'Pest and Disease Management:',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Text(
                    'Monitoring: Regularly inspect plants for signs of pests and diseases.\n'
                    'Integrated Pest Management (IPM): Implement IPM strategies, including biological control, cultural practices, and targeted pesticide application, to minimize pest damage.\n'
                    'Disease Prevention: Rotate crops, use disease-resistant varieties, and practice good sanitation to prevent the spread of diseases.\n',
                    style: TextStyle(fontSize: 16),
                  ),
                  SizedBox(height: 10),
                  Text(
                    'Growth and Development:',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Text(
                    'Vegetative Stage: Rice plants grow vegetatively, developing leaves and tillers.\n'
                    'Reproductive Stage: Flowering occurs, followed by the formation of grains.\n'
                    'Ripening Stage: Grains mature and change color from green to golden yellow.\n',
                    style: TextStyle(fontSize: 16),
                  ),
                  SizedBox(height: 10),
                  Text(
                    'Harvesting:',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Text(
                    'Timing: Harvest rice when the grains are fully mature but not overripe.\n'
                    'Methods: Harvesting methods include manual cutting with sickles, use of mechanical harvesters, or even combine harvesters in large-scale operations.\n',
                    style: TextStyle(fontSize: 16),
                  ),
                  SizedBox(height: 10),
                  Text(
                    'Post-Harvest Handling:',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Text(
                    'Threshing: Separate the grains from the stalks and husks using manual or mechanical threshers.\n'
                    'Drying: Dry the grains to reduce moisture content and prevent mold growth. This can be done on a drying floor or using mechanical dryers.\n'
                    'Storage: Store dried rice in clean, dry conditions to prevent spoilage and pest infestation.\n',
                    style: TextStyle(fontSize: 16),
                  ),
                  SizedBox(height: 10),
                  Text(
                    'Marketing and Distribution:',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Text(
                    'Packaging: Package rice into bags or containers suitable for transportation and sale.\n'
                    'Market Access: Sell rice through local markets, cooperatives, or directly to wholesalers and retailers.\n'
                    'Export: Some rice-producing regions export surplus rice to international markets, contributing to global trade.\n',
                    style: TextStyle(fontSize: 16),
                  ),
                  SizedBox(height: 10),
                  Text(
                    'Conclusion:',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Text(
                    'Throughout the entire process, careful management, including monitoring of weather conditions, pest and disease pressures, and soil fertility, is essential to ensure a successful rice harvest. Additionally, adapting practices to local conditions and embracing sustainable farming techniques can help optimize yields while minimizing environmental impacts.\n',
                    style: TextStyle(fontSize: 16),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
