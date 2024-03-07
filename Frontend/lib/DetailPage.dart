import 'package:flutter/material.dart';
import 'Buttons/RiceButton.dart'; // Import the RiceButton widget
import 'Buttons/MaizeButton.dart'; // Import the MaizeButton widget
import 'Buttons/ChickpeaButton.dart'; // Import the ChickpeaButton widget
import 'Buttons/MothbeansButton.dart'; // Import the MothbeansButton widget
import 'Buttons/KidneybeansButton.dart'; // Import the KidneybeansButton widget
import 'Buttons/PigeonpeasButton.dart'; // Import the PigeonpeasButton widget
import 'Buttons/MungbeanButton.dart';
import 'Buttons/BlackgramButton.dart';

import 'Buttons/LentilButton.dart'; // Import the RiceButton widget
import 'Buttons/PomegranateButton.dart'; // Import the MaizeButton widget
import 'Buttons/BananaButton.dart'; // Import the ChickpeaButton widget
import 'Buttons/MangoButton.dart'; // Import the MothbeansButton widget
import 'Buttons/GrapesButton.dart'; // Import the KidneybeansButton widget
import 'Buttons/WatermelonButton.dart'; // Import the PigeonpeasButton widget
import 'Buttons/MuskmelonButton.dart';
import 'Buttons/AppleButton.dart';


class DetailPage extends StatelessWidget {
  const DetailPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Detail About Crops'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(8.0),
        child: Scrollbar(
          child: ListView(
            children: [
              RiceButton(), // Use the RiceButton widget here
              SizedBox(height: 8), // Add spacing between buttons
              MaizeButton(), // Use the MaizeButton widget here
              SizedBox(height: 8), // Add spacing between buttons
              ChickpeaButton(), // Use the ChickpeaButton widget here
              SizedBox(height: 8), // Add spacing between buttons
              KidneybeansButton(), // Use the KidneybeansButton widget here
              SizedBox(height: 8), // Add spacing between buttons
              PigeonpeasButton(), // Use the PigeonpeasButton widget here
              SizedBox(height: 8), // Add spacing between buttons
              MothbeansButton(), // Use the MothbeansButton widget here
              SizedBox(height: 8),
              MungbeanButton(), // Use the MothbeansButton widget here
              SizedBox(height: 8),
              BlackgramButton(),
              SizedBox(height: 8),

              LentilButton(), // Use the RiceButton widget here
              SizedBox(height: 8), // Add spacing between buttons
              PomegranateButton(), // Use the MaizeButton widget here
              SizedBox(height: 8), // Add spacing between buttons
              BananaButton(), // Use the ChickpeaButton widget here
              SizedBox(height: 8), // Add spacing between buttons
              MangoButton(), // Use the KidneybeansButton widget here
              SizedBox(height: 8), // Add spacing between buttons
              GrapesButton(), // Use the PigeonpeasButton widget here
              SizedBox(height: 8), // Add spacing between buttons
              WatermelonButton(), // Use the MothbeansButton widget here
              SizedBox(height: 8),
              MuskmelonButton(), // Use the MothbeansButton widget here
              SizedBox(height: 8),
              AppleButton(),
              SizedBox(height: 8),


              // Repeat this pattern for all other buttons...
            ],
          ),
        ),
      ),
    );
  }
}
