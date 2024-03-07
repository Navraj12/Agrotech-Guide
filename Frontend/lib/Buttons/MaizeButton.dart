import 'package:flutter/material.dart';

class MaizeButton extends StatelessWidget {
  const MaizeButton({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      onPressed: () {
        // Navigate to MaizePage when button is pressed
        Navigator.push(
          context,
          MaterialPageRoute(builder: (context) => MaizePage()),
        );
      },
      style: ElevatedButton.styleFrom(
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(10), // Rectangular shape
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Text('Maize'),
      ),
    );
  }
}

class MaizePage extends StatelessWidget {
  const MaizePage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Maize Details'),
      ),
      body: SingleChildScrollView(
        child: Column(
          children: <Widget>[
            Image.asset(
              'assets/maize.png', // Path to your maize image
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
                    'Selection of Land: Choose a well-drained field with good soil fertility and access to sunlight.\n'
                    'Clearing: Clear the land of any existing vegetation, rocks, and debris.\n'
                    'Plowing and Tilling: Use plows or tillers to break up the soil and create a suitable seedbed. This improves soil aeration and water infiltration.\n',
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
                    'Variety Selection: Choose maize varieties suited to local climate, soil type, and intended use (e.g., grain production, silage, or forage).\n'
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
              'Timing: Plant maize when soil temperatures reach around 50-55°F (10-13°C), typically in spring.\n'
              'Spacing: Plant seeds in rows with appropriate spacing between plants, depending on the variety and desired plant population.\n'
              'Depth: Plant seeds at a depth of 1-2 inches (2.5-5 cm) in well-prepared soil.\n'
              'Planting Methods: Maize can be planted using mechanical planters, seed drills, or manually by hand or with dibblers.\n',
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
              'Soil Testing: Conduct soil tests to determine nutrient levels and pH, then apply fertilizers accordingly.\n'
              'Fertilizer Application: Apply fertilizers containing nitrogen, phosphorus, and potassium either before planting or as side-dressings during the growing season.\n'
              'Weed Control: Use pre-emergent herbicides, mechanical cultivation, or hand weeding to control weeds that compete with maize for nutrients, water, and sunlight.\n',
              style: TextStyle(fontSize: 16),
            ),
            SizedBox(height: 10),
            Text(
              'Irrigation:',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            Text(
              'Rainfed Cultivation: In areas with adequate rainfall, maize may rely solely on natural precipitation.\n'
              'Irrigation: In regions with insufficient rainfall, provide supplemental irrigation using techniques such as furrow, drip, or overhead irrigation.\n',
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
              'Monitoring: Regularly inspect maize plants for signs of pests, such as insects, rodents, and birds, as well as diseases like rust, smut, and blight.\n'
              'Integrated Pest Management (IPM): Employ a combination of cultural practices, biological control, and targeted pesticide applications to manage pests and diseases effectively while minimizing environmental impact.\n',
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
              'Vegetative Stage: Maize plants undergo vegetative growth, developing leaves, stems, and roots.\n'
              'Reproductive Stage: Tassel and ear development occur, followed by silk emergence and pollination.\n'
              'Grain Fill Stage: Kernels develop and fill out on the cob, requiring adequate moisture and nutrient availability.\n',
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
              'Timing: Harvest maize when kernels are fully mature but still moist, typically 20-35 days after silking.\n'
              'Methods: Harvest maize using combine harvesters for large-scale operations or manually with machetes or corn pickers for smaller fields.\n'
              'Storage: Store harvested maize in well-ventilated bins or silos to prevent spoilage from mold, insects, and rodents.\n',
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
              'Drying: Dry harvested maize to reduce moisture content to safe storage levels (usually around 13-15%).\n'
              'Shelling: Remove kernels from cobs using mechanical shellers or hand-operated devices.\n'
              'Cleaning: Remove debris, broken kernels, and foreign material from the maize using air screen cleaners or gravity tables.\n'
              'Grading and Packaging: Grade maize based on quality criteria such as kernel size, color, and moisture content. Package maize in sacks or bulk containers for storage or sale.\n',
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
              'Market Access: Sell maize through local markets, cooperatives, or directly to grain elevators, livestock feed mills, or food processors.\n'
              'Export: Some maize-producing regions export surplus grain to international markets, contributing to global trade.\n',
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
              'Throughout the entire process, careful management practices, including soil fertility management, pest and disease control, and irrigation management, are essential to ensure a successful maize harvest. Adaptation to local conditions and sustainable farming practices can help optimize yields while minimizing environmental impacts.',
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
