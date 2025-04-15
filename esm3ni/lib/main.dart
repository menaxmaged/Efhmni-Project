import 'package:flutter/material.dart';
import 'package:esm3ni/screens/home/home.dart';
import 'package:esm3ni/services/api-handler.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
 // await ApiHandler.getNgrokLink();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'esm3ni',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.blue,
        fontFamily: "Cairo",
      ),
      // home: VideoCaptureScreen(),
      // home: VideoCaptureExample(),
      home: const HomeScreen(),
    );
  }
}
