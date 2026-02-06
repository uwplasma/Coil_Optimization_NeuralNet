# Coil_Optimization_NeuralNet

Stellarator Coil Optimization with Neural Networks

A PyTorch-based neural network that predicts stellarator coil properties (arc length and mean squared curvature) from Fourier series coefficients. The goal is to eventually find the most optimal coils that have low magnetic flux errors and are feasibly engineerable. 

## Overview

This project uses deep learning to rapidly evaluate stellarator fusion reactor coil designs. Given Fourier coefficients describing a coil's 3D geometry, the model predicts:
- **Arc Length**: Total length of the coil (meters)
- **Mean Squared Curvature (MSC)**: Average squared curvature (1/m²)

Other objective functions like magnetic flux error, curvature, max_length, etc, are coming soon. 

This enables fast coil design iteration without expensive numerical simulations using SIMSOPT.

## Installation
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate   # On Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** If `simsopt` installation fails due to C++ compilation issues, try: `pip install simsopt==1.6.5`

### Training

Train a model from scratch or using saved data:
```bash
python3 train.py                    # Generate new random coils
python3 train.py saved_data.npz     # Train on existing data
```

Optional command-line arguments for data generation (see code comments for details).

### Evaluation

Evaluate the trained model:
```bash
python3 evaluate.py
```

Reports overall accuracy metrics and accuracy binned by coil length ranges.

## Model Architecture

- **Input**: 63 Fourier coefficients w/padding
- **Output**: 2 values [arc length, mean squared curvature]
- **Architecture**: 4-layer fully connected network (512 → 256 → 128 → 2 neurons)
- **Loss**: Weighted MSE (MSC weighted 20x higher than length)

See `train.py` for full architecture details.

## Data Format

Training data is stored in a file called `saved_data.npz`:
- `inputs`: Fourier coefficients, shape (N, 63)
- `labels`: [arc length, MSC], shape (N, 2)

Normalization statistics is saved in a file called `scaling_stats.npz` for consistent evaluation.

## Next Steps

### Data Improvements
- **QUASR Integration**: Use 50% real coils from QUASR database, 50% variations
- **Data Augmentation**: Use Fourier coefficients by ~1 standard deviation to generate realistic variants

### Accuracy Improvements
Currently, the neural network can predict accuracy with ~97% accuracy and mean squared curvature with ~22%. 
Ways to improve that could include better data (with real-life coils rather than randomly generated ones) or with 
different architectures or training procedures. This could also just be a hard problem to solve due to curvature's 
high variance and unpredictability. 

### Architecture Experiments
Experiment with some of the following neural network architecture and training choices for better accuracy (particularly on MSC): 
- Vary network depth (layers) and width (neurons per layer)
- Test different architectures: residual connections, bottleneck layers
- Experiment with activation functions: LeakyReLU, ELU, GELU, Swish
- Add regularization: dropout, batch normalization, L1/L2 weight decay
- Try different optimizers: SGD with momentum, RMSprop, AdamW
- Implement learning rate schedules: ReduceLROnPlateau, CosineAnnealing
- Explore separate gradient descent for length vs MSC predictions

### Error Analysis
- **Outlier Detection**: Generate box and whisker plots to identify problematic predictions
- **Pattern Recognition**: Analyze which coil geometries/coefficients lead to poor accuracy
- **Distribution Analysis**: Plot histograms of prediction errors to understand model behavior

## File Structure
```
Coil_Optimization/
├── train.py              # Model definition and training
├── evaluate.py           # Model evaluation and metrics
├── requirements.txt      # Python dependencies
└── README.md
```

The following three files are created during the training process and haven't been pushed to GitHub. 
├── saved_model.pt        # Trained model weights
├── saved_data.npz        # Training data
├── scaling_stats.npz     # Normalization parameters

## Requirements

- Python 3.9+
- PyTorch
- NumPy
- SIMSOPT (for coil geometry generation)

See `requirements.txt` for specific versions.
