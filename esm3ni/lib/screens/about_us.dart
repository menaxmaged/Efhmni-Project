import 'package:flutter/material.dart';
import 'package:esm3ni/shared/shared_widgets.dart';

class AboutUsScreen extends StatefulWidget {
  const AboutUsScreen({Key? key}) : super(key: key);

  @override
  State<AboutUsScreen> createState() => _AboutUsScreenState();
}

class _AboutUsScreenState extends State<AboutUsScreen> {
  final List<String> teamMembers = [
    "Mena Maged",
    "Marwa Salem",
    "Abdelrahman Ali Maher",
    "Abdelrahman Ahmed Goda",
    "Menna Khaled",
    "Sabry Salah",
    "Yousra Abdelzaher",
  ];

  @override
  void initState() {
    super.initState();
   // teamMembers.shuffle();
  }

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [
              Color(0xFF0E092A),
              Color(0xFF261D5C),
              Color(0xFF0E092A),
            ],
            begin: Alignment.centerLeft,
            end: Alignment.centerRight,
          ),
        ),
        child: Padding(
          padding:
              const EdgeInsets.only(right: 20, left: 20, top: 20, bottom: 10),
          child: ListView(
            physics: const BouncingScrollPhysics(),
            children: [
              Image.asset(
                "assets/images/app_logo.png",
                height: 200,
              ),
              const SizedBox(height: 20),
              const Text(
                "أسمعني هو تطبيق يهدف لترجمة لغة الصم و البكم المصرية لتسهيل التواصل بين الصم و البكم و بقية أفراد المجتمع مما يساعد فى زيادة اختلاط الصم و البكم بالمجتمع مستفيدين من قدراتمه و امكانياتهم ",
                style: TextStyle(color: Colors.white),
                maxLines: 6,

              ),
              Column(
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.end,
                    children: const [
                      Text(
                        "فريق أسمعنى",
                        style: TextStyle(fontSize: 24, color: Colors.white),
                      ),
                    ],
                  ),
                  const SizedBox(height: 20),
                  Column(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                        children: [
                          memberbox(memberName: teamMembers[0]),
                          memberbox(memberName: teamMembers[1])
                        ],
                      ),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                        children: [
                          memberbox(memberName: teamMembers[2]),
                          memberbox(memberName: teamMembers[3])
                        ],
                      ),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                        children: [
                          memberbox(memberName: teamMembers[4]),
                          memberbox(memberName: teamMembers[5]),
                        ],
                      ),
                    ],
                  )
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
