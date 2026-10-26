# ChronosAI Companion 🩺⏱️
An intelligent, edge-based vision-and-voice agentic system designed for vulnerable individuals living alone, powered by OpenCV 5.0.0 and AWS Serverless Cloud Infrastructure.

This project is fully developed using **VS Code**. It features a static front-end UI preview integrated with a Python/OpenCV Core and AWS Cloud capabilities.

---

## 🚀 How Judges Can Run and Test This Project Locally

Dear Judges, to try out this project on your local machine using VS Code, please follow these steps:

### 1. Clone & Setup Repository
Open your VS Code terminal and clone the repository:
```bash
git clone https://github.com
cd Chronos-AI-Companion
```

### 2. Download Face Tracking Models
Fetch the required YuNet model weights onto your local system:
```bash
python download_models.py
```

### 3. Install Pinned Dependencies
Install all the low-level sounddevice and OpenCV 5 requirements listed in the file:
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a file named `.env` in the root directory of the project and add your respective credentials:
```env
SENDER_EMAIL=your-email@gmail.com
RECEIVER_EMAIL=doctor-email@gmail.com
AWS_ACCESS_KEY_ID=your-aws-access-key-id
AWS_SECRET_ACCESS_KEY=your-aws-secret-access-key
AWS_REGION=ap-southeast-2
```

### 5. Run the Project Logic via VS Code Terminal
To launch the real-time high-performance OpenCV face detection pipeline and start the telemetry routing, trigger the main core execution controller file:
```bash
python chronos_companion.py
```

---
## 🌐 Project Links
* **Live UI Dashboard Preview (GitHub Pages):** [Insert your GitHub Pages Link Here]
* **Working Video Simulation Demo:** [Insert your Google Drive or YouTube Video Link Here]
