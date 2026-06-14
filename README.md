# 🌐 Smart Supply Chain AI Engine

An advanced, production-ready **FastAPI** microservice engine hosting a trained machine learning model for dynamic supply chain risk prediction and operational telemetry intelligence. Engineered for real-time risk classification and automated logistics routing optimization.

---

## 🚀 Key Features

- **Real-time Predictive Analytics:** Uses a trained Machine Learning pipeline to predict supply chain disruptions and delivery risks instantaneously.
- **Production-Grade API Infrastructure:** Built with FastAPI, utilizing asynchronous request handling and Pydantic validation for high-throughput operational intelligence.
- **Automated Telemetry Processing:** Features strict data validation and automated pre-processing layers to match model feature dependencies seamlessly.
- **Cloud Architecture & Resilience:** Implements dynamic on-demand model streaming from secure cloud storage to bypass standard repository constraints while maintaining memory efficiency during deployment.

---

## 🛠️ Tech Stack & Engineering Toolkit

| Category         | Technology            | Purpose                                                  |
| :--------------- | :-------------------- | :------------------------------------------------------- |
| **Framework**    | FastAPI               | High-performance asynchronous REST API architecture      |
| **Language**     | Python                | Core backend programming & data manipulation             |
| **ML Inference** | Scikit-learn / Pickle | Predictive pipeline and model tracking serialized states |
| **Validation**   | Pydantic v2           | Robust runtime data parsing and semantic request schemas |
| **Server**       | Uvicorn               | ASGI production server configuration                     |

---

## 📦 System Architecture Overview

The microservice decouples the heavyweight predictive model from the source control layer, ensuring a lightweight and scalable container footprint:

```text
       [ Client Request ]
               │  (JSON Payload via POST /predict)
               ▼
   ┌───────────────────────┐
   │  Pydantic Validation  │ ──(Enforces operational schema)
   └───────────────────────┘
               │
               ▼
   ┌───────────────────────┐
   │  Dynamic Load Check   │ ──(Downloads ML Model from Cloud Storage if absent)
   └───────────────────────┘
               │
               ▼
   ┌───────────────────────┐
   │   Inference Engine    │ ──(Runs Model Pipeline & Features Transformation)
   └───────────────────────┘
               │
               ▼
      [ JSON Response ]  ──(Risk Level, Delayed Probability & Optimizations)
```
