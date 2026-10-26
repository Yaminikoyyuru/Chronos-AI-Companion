# ChronosAI Companion 🩺⏱️

A vision-and-voice agentic companion for elders living alone. OpenCV 5 runs on the local computer
to detect a possible fall, stillness, or the person leaving the frame. The agent then asks
"Are you okay?" and, if there is no answer, alerts a physician through AWS (S3, SES, CloudWatch Logs).
The physician approves or rejects dispatch on a web page, and the robot responds out loud.
It also keeps the person company with quizzes, songs, news and routine reminders.

Competition entry: OpenCV 5 + AWS, Agentic Vision path.

## Links
* Demo video (under 5 minutes): <PASTE YOUTUBE UNLISTED LINK>
* Technical report: `report/report.pdf`
* Static UI preview (GitHub Pages, not connected to the live system): <PASTE LINK OR DELETE THIS LINE>

## Tested on
Windows, Python 3.14, `opencv-python` 5.0.0, webcam and microphone.
It also runs without a webcam, microphone, speakers, or AWS keys (see below).

## How judges can run it (no AWS account needed)

```
git clone https://github.com/<your-username>/Chronos-AI-Companion.git
cd Chronos-AI-Companion
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python download_models.py
python chronos_companion.py --demo
```

Without AWS keys the app runs in LOCAL mode: every decision is written to `agent_trace.jsonl`
and nothing is sent to the cloud. (Linux/macOS: use `source .venv/bin/activate`.)

No webcam? Run on a recorded clip (loops):
```
python chronos_companion.py --video sample_clips/immobile_01.mp4 --skip-onboarding --immobile 8
```

## What to try

| Action | Expected result |
|---|---|
| Sit still in front of the camera | After the timer the robot asks "Are you okay?" |
| Say "I am fine" (or press **y**) | Alert is cancelled (`cancel_alert` in the trace) |
| Do nothing for two questions | Escalation (`escalate_to_physician`) |
| Press **f** (simulated fall) | Different question ("Are you hurt?"), shorter wait |
| Say "I need a break", then "I am back" | Monitoring pauses and resumes; "help" always overrides |

Hotkeys (click the robot window first): `s` songs, `g` quiz, `n` news, `e` distress, `f` fall,
`i` immobile, `y` "I am fine", `h` "help", `r` reset, `q` quit.

## Evaluation
```
python evaluate.py --dir sample_clips --immobile 8 --facelost 5
```
Record your own clips with `python record_clip.py normal_01 18`.

## Optional: full AWS mode (your own account)
Create `.env` from `.env.example`. No keys are included in this repository.

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=ap-southeast-2
S3_BUCKET=your-bucket
STORE=s3
EMAIL_BACKEND=ses
SENDER_EMAIL=your-verified-email@gmail.com
RECEIVER_EMAIL=doctor-email@gmail.com
```
S3 holds the snapshot, incident record, physician review page and the physician's decision.
SES sends the alert email. CloudWatch Logs receives the agent trace.

## Limitations
Fall detection is a heuristic, not a trained model. The evaluation set is small (one person).
Voice uses Google speech recognition (internet needed, English only). On our AWS account
DynamoDB, Lambda, SNS and Bedrock were unavailable, so we use S3 and rule-based risk.
This is a research prototype, not a medical device.
