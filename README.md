# PINNs: A Scalable Physics-Informed Neural Network Suite

This repository contains a modular and scalable implementation of **Physics-Informed Neural Networks (PINNs)** using PyTorch. Unlike standard neural networks that rely solely on data, PINNs embed the underlying partial differential equations (PDEs) into the loss function, allowing the model to learn the physics of the system.

## 🚀 Features

* **Equation Agnostic**: Easily switch between 1D Burgers, 1D Heat, and 2D Heat equations via configuration files.
* **Modular Architecture**: Separate directories for core solver logic, physics definitions, and visualization utilities.
* **Automatic Differentiation**: Uses PyTorch `autograd` to compute exact derivatives for the PDE residuals.
* **Customizable**: Control network depth, width, and learning rates through YAML files in the `config/` folder.

---

## 📂 Project Structure

```text
PINNs/
├── core/
│   ├── networks.py      # Flexible MLP architectures
│   └── pinn_solver.py   # Training engine & derivative calculations
├── problems/            # Physics definitions (PDE residuals & data)
│   ├── burgers_1D.py    
│   └── heat_2D.py       
├── utils/
│   └── plotting.py      # Heatmaps and loss curve visualization
├── config/              # Hyperparameters for different runs
│   ├── burgers.yaml
│   └── heat_2d.yaml
├── main.py              # Entry point for the framework
└── requirements.txt     # Python dependencies

```

---

## 🧠 Physics Covered

### 1. 1D Burgers' Equation

The Burgers' equation is a fundamental PDE in fluid mechanics:



It is used to model the formation and propagation of shock waves.

### 2. 2D Heat Equation

Models how heat diffuses through a 2D plane over time:


---

## 🛠️ Installation & Usage

### 1. Clone the repo

```bash
git clone https://github.com/gulzarali19/PINNs.git
cd PINNs

```

### 2. Install dependencies

```bash
pip install -r requirements.txt

```

### 3. Run a simulation

To train the model using a specific configuration (e.g., Burgers' Equation):

```bash
python main.py --config config/burgers.yaml

```

---

## 📊 Results

The solver generates heatmaps showing the predicted evolution of the system.

* **1D Problems**: Time () vs Space () heatmaps.
* **2D Problems**: Spatial snapshots () at various time intervals.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

