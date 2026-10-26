# ChronosAI Companion 🩺⏱️

An intelligent, edge-based vision-and-voice agentic system designed for vulnerable individuals living alone, powered by **OpenCV 5.0.0** and **AWS Serverless Cloud Infrastructure**.

## 🚀 Core Features
* **Active Perception Pipeline:** Leverages advanced `cv2.FaceDetectorYN` (YuNet) array tracking combined with Dense Farneback Optical Flow heuristics to measure real-time patient movement dynamics.
* **Intelligent Agentic Loop:** Implements a strict time-bound agentic state machine that executes multi-stage verification cooldown loops to mitigate annoying false alarms.
* **Contextual Cooldown Safeguards:** Respects human privacy boundaries with native voice commands like "I need a break" or "good night", while instantly overriding pauses if the user requests "help".
* **Serverless Cloud Telemetry:** Eliminates sandbox account SCP blockages by securely processing structural logs via serverless **AWS S3 Select** SQL queries directly out of an S3 hosted transactional `agent_trace.jsonl` stream.
* **Enterprise Logging & Alerting:** Continuously tracks operational metrics, agent transitions, and system triggers via **Amazon CloudWatch Logs** with an automated **7-day privacy-first S3 Lifecycle destruction policy**, while handling instant doctor notifications through **Amazon SES**.

## 📁 Repository Directory
```text
chronos/
├── .gitignore
├── requirements.txt                         # Low-level sounddevice & OpenCV 5 dependencies
├── perception.py                        # YuNet tracking & Optical Flow heuristics engine
├── robot_ui.py                          # GUI human-agent interaction loop matrix
├── chronos_companion.py                 # Main Agentic State Machine loop controller
├── download_models.py                   # On-device runtime weights verification fetch script
├── agent_trace.jsonl                    # Local database stream analyzed later via S3 Select
├── incidents_local.jsonl                # Temporary local anomaly storage arrays
├── aws.py                               # Low-level AWS SDK credentials setup
├── cloud.py                             # S3 Select query triggers, SES routes & CloudWatch
└── testmail.py                          # Independent SMTP alert test suite routine
```

## 🛠️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone <your-github-repo-link>
   cd chronos
   ```

2. **Download Face Tracking Models:**
   ```bash
   python download_models.py
   ```

3. **Install Pinned Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Setup:**
   Create a `.env` file in the root directory and update your local variables:
   ```env
   SENDER_EMAIL=your-email@gmail.com
   RECEIVER_EMAIL=doctor-email@gmail.com
   AWS_ACCESS_KEY_ID=your-aws-access-key-id
   AWS_SECRET_ACCESS_KEY=your-aws-secret-access-key
   AWS_REGION=ap-southeast-2
   ```

5. **Run the Edge Companion Agent:**
   ```bash
   python chronos_companion.py
   ```

## 📊 System Architecture & Workflow

```text
[Patient / Camera Scene] ➔ [perception.py (YuNet + Flow)] ➔ [chronos_companion.py Engine]
                                                                     │
                                                           (If Anomaly Detected)
                                                                     ▼
                                                       [Background English Voice Check-In]
                                                                     │
                                     ┌───────────────────────────────┴───────────────────────────────┐
                                     ▼ (Unresponsive / Distress)                                     ▼ (Responded "I am fine")
                         [Action: Escalate to AWS Cloud]                                             [Action: Reset Idle Timers]
                                     │
                                     ├───────────────────────────────┐
                                     ▼                               ▼
                       [S3 Trace Upload + CloudWatch]       [Amazon SES Email Alert]
                                                                     │
                                                                     ▼
                                                       [Live Physician Portal Dashboard]
```
