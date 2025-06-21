# 👁️‍🗨️ Vision Aid – ESP32-CAM Based Assistive System

A smart **vision assistance system** powered by **ESP32-CAM**, designed to aid visually impaired individuals by providing real-time image capture and processing, audio feedback, and object detection.

---

## 💡 Project Overview

This project leverages the low-cost **ESP32-CAM** module to capture images and send them for processing. Based on the analysis (e.g., obstacle detection, object recognition), appropriate **audio feedback** is provided to assist the user in navigating or understanding their environment.

---

## 🔧 Hardware Components

- 📷 **ESP32-CAM** module
- 🔌 USB to TTL (FTDI) programmer
- 🔊 Optional: Speaker or Buzzer
- 🔋 5V Power Supply (battery or USB)
- 🧠 Edge or cloud ML server (for processing, if used externally)

---

## 🧰 Software & Tools

- Arduino IDE / PlatformIO
- ESP32 Board Package
- Flask / Python (for server-side image processing if applicable)
- Google Text-to-Speech (gTTS) or similar for audio feedback

---

## 🛠️ Setup & Installation

1. **Install ESP32 Boards in Arduino IDE**
   - File > Preferences > Add board manager URL:  
     `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`

2. **Connect ESP32-CAM to FTDI Module**
   - GPIO0 to GND for flashing
   - Power via 5V & GND
   - TX/RX swapped between ESP32 and FTDI

3. **Upload Code**
   - Use Arduino or PlatformIO to upload your sketch (`.ino` or `.cpp`)
   - Remove GPIO0 from GND after upload for normal boot

4. **Run Python Flask Server (optional)**
   - If using a server to process images and generate alerts:
     ```bash
     pip install flask opencv-python gtts
     python app.py
     ```

---

## 🔁 Working Principle

1. ESP32-CAM captures an image periodically or on-demand.
2. The image is either:
   - Processed onboard (basic edge detection / QR / motion sensing)
   - OR sent to a server for ML-based object detection.
3. Based on results, audio feedback is played:
   - Example: “Obstacle ahead”, “Person detected”, “Door on right”

---

## 🎯 Applications

- Obstacle detection and navigation aid
- Smart assistant for blind or low-vision individuals
- Indoor guidance system
- Voice alerts for detected objects or scenes

---




## 📜 License

This project is open-source under the **MIT License**.

---

## 🙋‍♀️ Author

Developed with purpose and passion by **[Eniya Sre A G](https://github.com/Eniyasre-AG)**  
Empowering lives through tech 🤖✨

---

## 🔧 Future Enhancements

- Integrate TensorFlow Lite for onboard object detection
- Add haptic feedback for alerts
- Improve low-light image handling
- Add voice control (wake-word detection)
