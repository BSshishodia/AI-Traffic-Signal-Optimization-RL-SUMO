# 🚦 AI Traffic Signal Optimization using Reinforcement Learning & SUMO

---

## 🧠 Project Overview

This project presents an **AI-based adaptive traffic signal control system** using **Reinforcement Learning (Q-Learning)** and **SUMO (Simulation of Urban Mobility)**.

The system dynamically adjusts traffic signals based on real-time traffic conditions to **reduce congestion, waiting time, and improve traffic flow**.

---

## 🎯 Objectives

* 🚗 Minimize traffic congestion (queue length)
* ⏱ Reduce vehicle waiting time
* 🧠 Develop an intelligent adaptive traffic system
* 📊 Compare RL-based system with traditional fixed-time signals

---

## ⚙️ Tech Stack

* **Python**
* **SUMO (Traffic Simulation)**
* **Reinforcement Learning (Q-Learning)**
* **PyTorch (LSTM - Traffic Prediction)**
* **Pandas, NumPy**
* **Matplotlib (Visualization)**

---

## 🏗️ Project Architecture

```
Traffic Simulation (SUMO)
        ↓
   State (Queue Length)
        ↓
Reinforcement Learning Agent
        ↓
Traffic Signal Control
        ↓
Improved Traffic Flow
```

---

## 🤖 Models Used

### 🔹 Reinforcement Learning (Q-Learning)

* Learns optimal signal switching policy
* Uses reward = **-queue length**
* Improves over episodes

---

### 🔹 LSTM Model (Support Module)

* Trained on traffic dataset (J12 junction)
* Predicts future traffic queue
* Used for analysis (not directly integrated with RL)

---

## 📊 Results & Analysis

### 🔹 RL Learning Curve

* Shows improvement from random decisions → optimized control

### 🔹 Comparison

| System          | Performance          |
| --------------- | -------------------- |
| Fixed Signal    | High congestion ❌    |
| RL-Based Signal | Reduced congestion ✅ |

---

## 📈 Visualizations

### 🔸 Queue Comparison

* RL significantly reduces queue length

### 🔸 Vehicle Flow

* Smoother traffic movement with RL

---

## 📁 Project Structure

```
TDLP/
│
├── analysis/        # Graphs and evaluation
├── models/          # LSTM model
├── rl/              # RL environment & agent
├── sumo_env/        # SUMO config files
├── assets/          # Images/screenshots
│
├── main.py
├── preprocess.py
├── requirements.txt
├── README.md
```

---

## ▶️ How to Run

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Train RL Model

```bash
python rl/train_rl.py
```

---

### 3️⃣ Run Baseline (Without RL)

```bash
python rl/evaluate_rl.py
```

---

### 4️⃣ Run RL Model

```bash
python rl/evaluate_rl_with_model.py
```

---

### 5️⃣ Visualize Results

```bash
python analysis/compare.py
python analysis/plot.py
```

---

## 🚀 Key Features

✔ Adaptive traffic signal control
✔ Real-time learning using RL
✔ Traffic prediction using LSTM
✔ Performance comparison with baseline
✔ Clean modular architecture

---

## 🧠 Key Insight

> Instead of fixed-time signals, an RL-based system can dynamically adapt to traffic conditions, significantly improving efficiency.

---

## 📌 Future Improvements

* 🔄 Multi-junction optimization
* 🧠 Deep RL (DQN / PPO)
* 🌍 Real-world traffic dataset integration
* 📱 Web dashboard visualization

---

## 👨‍💻 Author

**BHAV SHISHODIA**
B.Tech & M.Tech in AI & ML

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and feel free to contribute!

---
