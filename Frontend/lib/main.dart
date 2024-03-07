import 'package:custom/APIpage.dart';
import 'package:flutter/material.dart';
import 'SoilCheckPage.dart'; // Importing the SoilCheckPage component
import 'HelpPage.dart'; // Importing the HelpPage component
import 'MorePage.dart'; // Importing the MorePage component
import 'DetailPage.dart'; // Importing the DetailPage component

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      debugShowCheckedModeBanner: false,
      initialRoute: '/', // Set initial route
      routes: {
        '/': (context) => MyHomePage(title: 'Flutter Demo Home Page'), // Home page
        '/soil_check': (context) => SoilCheckPage(), // Route for Soil Check Page
        '/help': (context) => HelpPage(), // Route for Help Page
        '/more': (context) => MorePage(), // Route for More Page\
        '/API': (context) => APIpage(), // Route for More Page\
        '/detail': (context) => DetailPage(), // Route for Detail Page
        

      },
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({Key? key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Color.fromARGB(255, 70, 156, 106),
        title: Text(
          'AGROTECHGUIDE',
          style: TextStyle(
            fontSize: 24,
            fontWeight: FontWeight.bold,
            color: Colors.white,
          ),
        ),
        centerTitle: true,
      ),
      drawer: Drawer(
        child: ListView(
          padding: EdgeInsets.zero,
          children: <Widget>[
            DrawerHeader(
              decoration: BoxDecoration(
                color: Color.fromARGB(255, 70, 156, 106),
              ),
              child: Text(
                'Menu',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 24,
                ),
              ),
            ),
            ListTile(
              title: Text('Help'),
              onTap: () {
                Navigator.pushNamed(context, '/help');
              },
            ),
            ListTile(
              title: Text('More'),
              onTap: () {
                Navigator.pushNamed(context, '/more');
              },
            ),
            ListTile(
              title: Text('API'),
              onTap: () {
                Navigator.pushNamed(context, '/API');
              },
            ),
          ],
        ),
      ),
      body: Stack(
        children: [
          Image.asset(
            'assets/b.jpg',
            width: double.infinity,
            fit: BoxFit.cover,
          ),
          Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                ElevatedButton(
                  onPressed: () {
                    Navigator.pushNamed(context, '/soil_check'); // Navigate to Soil Check Page
                  },
                  child: Text(
                    'Check Your Soil',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 16, // Increase font size
                    ),
                  ),
                  style: ElevatedButton.styleFrom(
                    primary: Color.fromARGB(255, 70, 156, 106),
                    minimumSize: Size(180, 60), // Increase button size
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(10), // Adjust the border radius as needed
                    ),
                  ),
                ),
                SizedBox(height: 20),
                ElevatedButton(
                  onPressed: () {
                    Navigator.pushNamed(context, '/detail'); // Navigate to Detail Page
                  },
                  child: Text(
                    'Crops Details',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 16, // Increase font size
                    ),
                  ),
                  style: ElevatedButton.styleFrom(
                    primary: Color.fromARGB(255, 70, 156, 106),
                    minimumSize: Size(180, 60), // Increase button size
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(10), // Adjust the border radius as needed
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
