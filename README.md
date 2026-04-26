# Lustra 🔥
### Vision-Based Wildfire Detection, Mapping, and Spread Prediction

**Lustra** is an computer vision pipeline designed to **detect, map, and predict wildfire spread** using aerial imagery and environmental data.

The system simulates drone-based monitoring and combines **computer vision, spatial analysis, and environmental modelling** to provide early wildfire intelligence.

The long-term goal of Lustra is to support **wildfire monitoring systems** capable of identifying fires and forecasting how they may evolve.

---

# Project Goals

Lustra is built around three main objectives.

## 1. Wildfire Detection

Detect fire in aerial imagery using deep learning models.

Approaches include:

- YOLO-based object detection  
- CNN-based image classification  
- Smoke/fire segmentation  

---

## 2. Wildfire Mapping

Localize detected fires in **3D space or geographic coordinates**.

This includes:

- Depth estimation  
- Stereo vision  
- Camera geometry  
- Spatial mapping  

The goal is to determine:

- Fire location  
- Fire area  
- Spatial distribution  

---

## 3. Wildfire Spread Prediction

Predict how the wildfire will evolve based on **environmental conditions**.

Potential inputs include:

- Wind speed  
- Wind direction  
- Terrain slope  
- Vegetation density  
- Humidity  
- Temperature  

These factors will be used to model **future fire spread patterns**.

---

# System Architecture

High-level pipeline of Lustra:

```
Drone / Simulation Environment
            │
            ▼
      Image Capture
            │
            ▼
    Fire Detection Model
            │
            ▼
     Depth Estimation
     (Stereo / AI)
            │
            ▼
     Fire Localization
            │
            ▼
    Environmental Data
            │
            ▼
     Spread Prediction
```

---

# Current Development

Current development focuses on building the **visual perception pipeline**.

Implemented components include:

- Drone simulation environment  
- Camera image capture  
- Depth estimation experiments  
- Stereo vision research  
- Dataset preparation  

Simulation is performed using **PyBullet** to replicate aerial observation scenarios.

---

# Technologies

Main tools used in this project:

- Python  
- OpenCV  
- PyTorch  
- YOLO  
- PyBullet  
- NumPy  
- Matplotlib  

Possible future integrations:

- Weather APIs  
- Reinforcement learning for fire modelling  

---

# Future Work

Planned improvements:

- Train wildfire detection models  
- Build geospatial fire mapping  
- Incorporate weather and terrain data  
- Implement fire spread simulation models  

---

# Motivation

Wildfires are becoming increasingly frequent and destructive due to climate change. Detection and accurate prediction of wildfire spread can significantly improve response strategies and reduce environmental damage.

Lustra explores how **computer vision and AI systems can assist wildfire monitoring and forecasting** using aerial data.

---

# Authors

```
Aras Fırat
Berat Bora Altaș
Tunda Demirci
```



Izmir University of Economics  
