---
title: Helmet Safety Dashboard
emoji: 🐢
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 5.38.0
app_file: app.py
pinned: false
license: mit
short_description: Smart helmet detection on construction sites
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
#  AI Helmet Safety Dashboard

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Shakhzod-Shohn/helmet-safety-dashboard)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-blueviolet)](https://github.com/ultralytics/ultralytics)
[![Gradio](https://img.shields.io/badge/%F0%9F%A gradio-ui-orange)](https://gradio.app)

An enterprise-grade, AI-powered web application for monitoring safety helmet compliance on construction sites. This system analyzes uploaded video footage to detect workers with and without helmets, providing detailed analytics, violation evidence, and historical data tracking.

**[View the Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/Shakhzod-Shohn/helmet-safety-dashboard)**

 


---

##  Features

- **Real-Time Object Detection:** Utilizes a custom-trained YOLOv8 model to detect 'helmet' and 'head' (no helmet) classes in video streams.
- **Interactive Web Dashboard:** A user-friendly interface built with Gradio for easy video upload and analysis.
- **Advanced Analytics:** Generates detailed statistics for each analysis, including total violations, severity breakdowns, and performance metrics.
- **Violation Evidence:** Automatically captures and displays image snapshots of frames where a safety violation is detected.
- **Detection Zones:** Users can define specific rectangular zones within the video frame to focus the analysis on critical areas.
- **Data Persistence:** All detected violations are saved to an SQLite database, enabling historical data review.
- **Historical Trends:** An interactive dashboard to visualize violation trends over time.
- **Alerting System:** Flags periods of high violation frequency based on a configurable threshold.

## Technology Stack

This project integrates a modern stack for machine learning and web deployment:

- **AI/ML:**
  - **Model:** YOLOv8s (You Only Look Once, version 8)
  - **Training:** Fine-tuned on a custom dataset of construction site images using Google Colab.
  - **Core Libraries:** PyTorch, Ultralytics, OpenCV
- **Backend & Data:**
  - **Database:** SQLite
  - **Data Manipulation:** Pandas, NumPy
- **Frontend & UI:**
  - **Framework:** Gradio
  - **Charting:** Plotly
- **Deployment:**
  - **Platform:** Hugging Face Spaces
  - **Version Control:** Git & Git LFS
