<h4>AI & Machine Learning Engineering</h4>


Welcome to my AI/ML project hub. This repository serves as a centralized portfolio showcasing production-ready implementations across Computer Vision, Predictive Analytics, and Intelligent Agentic Workflows.

Each subfolder contains an independent, fully functional project complete with its production history, architecture blueprints, and source code.

---

## 📊 Quick Portfolio Overview

| Project Folder | Core Technology | Primary AI Task | Business / Technical Impact |
| :--- | :--- | :--- | :--- |
| **[yolov8-object-detector](./yolov8-object-detector)** | YOLOv8, PyTorch, OpenCV | Real-Time Object Detection | High-accuracy tracking of pedestrians, vehicles, and assets in dynamic video feeds. |
| **[Prediction-Model](./Prediction-Model)** | Scikit-Learn, Pandas, XGBoost | Proactive Churn Prediction | Turns reactive customer support into proactive retention, creating an early warning window before cancellation. |
| **[Agentic-Project](./Agentic-Project)** | LangChain, LLMs, Vector DBs | Autonomous Email Agent | Automates end-to-end email parsing, categorization, listening, draft generation, and auto-replies. |

---

## 🛠️ Detailed Project Breakdowns

### 👁️ 1. Real-Time Object Detection Pipeline (YOLOv8)
* **Directory:** [`/yolov8-object-detector`](./yolov8-object-detector)
* **Objective:** Built a high-throughput, real-time computer vision system optimized for multi-class detection, specifically targeting vehicles (cars) and pedestrians (people) in complex visual environments.
* **Key Achievements:**
  * Configured transfer learning hooks on pre-trained YOLOv8 models to optimize inference speeds across multi-object frames.
  * Designed an end-to-end data processing workflow utilizing OpenCV to parse live stream inputs, handle frame resizing, and draw precise bounding boxes.
  * Structured deterministic evaluation loops to validate localization and classification accuracy.
* **Tech Stack:** Python, PyTorch, Ultralytics YOLOv8, OpenCV, NumPy.

### 📉 2. Proactive Customer Churn Prediction Engine
* **Directory:** [`/Prediction-Model`](./Prediction-Model)
* **Objective:** Developed an intelligent predictive analytics engine designed to transform reactive customer support into a proactive retention strategy by identifying high-risk subscribers before they cancel.
* **Key Achievements:**
  * Engineered a high-impact "early warning window" allowing Customer Success and Marketing teams strategic lead time to execute targeted interventions.
  * Developed robust data pre-processing scripts handling class imbalances, missing data imputation, and categorical encoding using Pandas and Scikit-Learn.
  * Optimized classification thresholds (focusing heavily on Precision-Recall curves and ROC-AUC) to minimize false negatives and guarantee highly targeted user alerts.
* **Tech Stack:** Python, Pandas, Scikit-Learn, XGBoost, Matplotlib, Seaborn.

### 🤖 3. Intelligent Autonomous Email Management Agent
* **Directory:** [`/Agentic-Project`](./Agentic-Project)
* **Objective:** Built an advanced LLM-powered multi-agent system designed to act as an automated customer communications hub by listening, parsing, categorizing, and replying to incoming messages.
* **Key Achievements:**
  * Architected structured cognitive patterns enabling the agent to ingest unstructured emails, extract intent, and categorize issues instantly.
  * Implemented an advanced reasoning loop that drafts contextually aware responses and securely interfaces with communication endpoints for automated replies.
  * Integrated system guardrails and contextual memory states to ensure generated email drafts remain strictly aligned with user constraints and brand tone.
* **Tech Stack:** Python, OpenAI API / Anthropic API, LangChain / CrewAI, Pydantic (Data Validation).

---

## 📈 System Engineering & Best Practices
Across all projects, I emphasize enterprise-grade machine learning and development principles:
* **Production-Level Modularity:** Code is cleanly separated into operational segments (e.g., data ingestion, model processing, evaluation, and runtime execution).
* **Hardened Security Architecture:** Implemented secure repository practices utilizing structural `.gitignore` configurations to completely isolate sensitive active API credentials (`token.json`) from the public domain.
* **Deterministic Tracking:** Full integration of Git history across all merged projects to demonstrate iterative code growth and professional development tracking.

---
📫 **Contact & Collaboration:** 
Feel free to explore the project codebases or reach out to me via the contact links on my main GitHub profile page!
