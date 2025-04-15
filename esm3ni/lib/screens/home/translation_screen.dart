import 'dart:async';
import 'dart:io';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:esm3ni/provider/provider.dart';
import 'package:esm3ni/services/api-handler.dart';
import 'package:esm3ni/shared/constants.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';
import 'package:video_player/video_player.dart';

class TranslationScreen extends StatefulWidget {
  const TranslationScreen({Key? key}) : super(key: key);

  @override
  State<TranslationScreen> createState() => _TranslationScreenState();
}

class _TranslationScreenState extends State<TranslationScreen> {
  String result = "";
  VideoPlayerController? _videoController;

  bool videoTranslated = false;
  bool cameraToCapture = false;
  bool videoLoaded = false;
  bool isUploading = false;

  late CameraController _cameraController;
  late List<CameraDescription> _cameras;
  int _countdownSeconds = 3;
  bool _isCountdownActive = false;
  bool videoCaptureStarted = false;
  XFile? file;

  @override
  void initState() {
    super.initState();
    _setupCamera();
  }

  @override
  void dispose() {
    _videoController?.dispose();
    _cameraController.dispose();
    super.dispose();
  }

  Future<void> _setupCamera() async {
    _cameras = await availableCameras();
    _cameraController = CameraController(_cameras[0], ResolutionPreset.high);
    await _cameraController.initialize();
    setState(() {});
  }

  void _startCountdown() {
    setState(() {
      _isCountdownActive = true;
    });
    Timer.periodic(const Duration(seconds: 1), (Timer timer) {
      if (_countdownSeconds > 1) {
        setState(() {
          _countdownSeconds--;
        });
      } else {
        timer.cancel();
        setState(() {
          _countdownSeconds = 3;
          _isCountdownActive = false;
        });
        _captureVideo();
      }
    });
  }

  Future<void> _captureVideo() async {
    setState(() {
      videoCaptureStarted = true;
    });
    await _cameraController.startVideoRecording();
  }

  Future<void> _stopVideo() async {
    setState(() {
      videoCaptureStarted = false;
      cameraToCapture = false;
      videoLoaded = true;
    });
    file = await _cameraController.stopVideoRecording();
    if (file != null) {
      await displayChosenVideo(path: file!.path);
    }
  }

  Future<void> displayChosenVideo({required String path}) async {
    setState(() {
      result = "";
      isUploading = true;
    });

    _videoController = VideoPlayerController.file(File(path));
    await _videoController!.initialize();
    _videoController!.play();
    _videoController!.setVolume(0);
    _videoController!.setLooping(true);

    setState(() {
      videoLoaded = true;
    });

    try {
      String? response = await ApiHandler.uploadVideo(filePath: path);
      setState(() {
        result = response?.trim().isEmpty ?? true
            ? "حدث خطأ أثناء التحميل."
            : response!;
      });
    } catch (_) {
      setState(() {
        result = "حدث خطأ أثناء التحميل.";
      });
    } finally {
      setState(() {
        isUploading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Consumer<MyProvider>(
      builder: (context, myModel, child) {
        return SafeArea(
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 15, vertical: 10),
            decoration: backgroundDecoration,
            child: Column(
              mainAxisAlignment: MainAxisAlignment.start,
              children: [
                appBar(),
                Expanded(
                  child: cameraToCapture
                      ? cameraCaptureWidget()
                      : videoLoaded && _videoController != null && _videoController!.value.isInitialized
                          ? displayVideo()
                          : noVideoWidget(),
                ),
                const SizedBox(height: 10),
                const Divider(thickness: 4, height: 4, color: Colors.lightGreenAccent),
                const SizedBox(height: 20),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    translationButton(
                      text: "ترجمة فيديو\n على الهاتف",
                      icon: Icons.upload,
                      onTap: isUploading
                          ? null
                          : () async {
                              final picker = ImagePicker();
                              final pickedFile = await picker.pickVideo(source: ImageSource.gallery);
                              if (pickedFile != null) {
                                await displayChosenVideo(path: pickedFile.path);
                              }
                            },
                    ),
                    translationButton(
                      text: "التقاط فيديو",
                      icon: Icons.videocam_outlined,
                      top: false,
                      onTap: isUploading
                          ? null
                          : () {
                              setState(() {
                                cameraToCapture = true;
                              });
                            },
                    ),
                  ],
                ),
                const SizedBox(height: 10),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget translationButton({
    required String text,
    required IconData icon,
    bool top = true,
    void Function()? onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Opacity(
        opacity: onTap == null ? 0.5 : 1.0,
        child: Container(
          width: 150,
          height: 150,
          decoration: BoxDecoration(
            gradient: LinearGradient(
              colors: const [
                Colors.lightGreenAccent,
                Color(0x00000000),
              ],
              begin: top ? Alignment.topCenter : Alignment.bottomCenter,
              end: top ? Alignment.bottomCenter : Alignment.topCenter,
            ),
            borderRadius: BorderRadius.circular(25),
            border: Border.all(width: 2.0, color: Colors.transparent),
          ),
          child: Container(
            decoration: BoxDecoration(
              color: Colors.black,
              borderRadius: BorderRadius.circular(25),
            ),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(icon, size: 60, color: Colors.white),
                Text(
                  text,
                  textAlign: TextAlign.center,
                  textDirection: TextDirection.rtl,
                  style: const TextStyle(fontSize: 20, color: Colors.white),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget loadingWidget() {
    return Container(
      padding: const EdgeInsets.all(10),
      width: 150,
      height: 60,
      decoration: BoxDecoration(
        color: Colors.yellow.withOpacity(0.7),
        borderRadius: BorderRadius.circular(25),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: const [
          Text("تحميل", style: TextStyle(color: Colors.white, fontSize: 24)),
          SizedBox(width: 10),
          CircularProgressIndicator(color: Colors.white),
        ],
      ),
    );
  }

  Widget displayVideo() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      children: [
        SizedBox(
          height: 250,
          child: AspectRatio(
            aspectRatio: _videoController!.value.aspectRatio,
            child: VideoPlayer(_videoController!),
          ),
        ),
        const SizedBox(height: 10),
        isUploading
            ? loadingWidget()
            : Container(
                padding: const EdgeInsets.all(10),
                width: 300,
                height: 60,
                decoration: BoxDecoration(
                  color: result == "عذرا, لم يتم التحقق جيدا من الجملة."
                      ? Colors.red
                      : Colors.lightGreenAccent.withOpacity(0.7),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Text(
                  result,
                  maxLines: 3,
                  textAlign: TextAlign.center,
                  style: const TextStyle(color: Colors.white, fontSize: 20),
                ),
              ),
      ],
    );
  }

  Widget noVideoWidget() {
    return Image.asset("assets/images/no-video-img.jpg");
  }

  Widget cameraCaptureWidget() {
    return Column(
      children: [
        if (_cameraController.value.isInitialized)
          SizedBox(
            height: 300,
            width: 300,
            child: CameraPreview(_cameraController),
          ),
        const SizedBox(height: 20),
        _isCountdownActive
            ? Text('$_countdownSeconds', style: const TextStyle(fontSize: 28, color: Colors.white))
            : Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  ElevatedButton(
                    onPressed: videoCaptureStarted ? _stopVideo : _startCountdown,
                    child: Icon(videoCaptureStarted ? Icons.pause : Icons.fiber_manual_record),
                  ),
                  const SizedBox(width: 10),
                  TextButton(
                    onPressed: () {
                      setState(() {
                        cameraToCapture = false;
                      });
                    },
                    child: const Text("إلغاء", style: TextStyle(color: Colors.white)),
                  ),
                ],
              ),
      ],
    );
  }

  Widget appBar() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        const SizedBox(),
        Image.asset("assets/images/app_logo.png", width: 100, height: 100),
      ],
    );
  }
}
