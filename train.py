import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from simsopt.geo import CurveXYZFourier, CurveLength, MeanSquaredCurvature
import random
from torch.utils.data import TensorDataset, DataLoader
import os
import sys


# Fourier series with maximum order of magnitude of 10
# Degrees of freedom of 63 (20 sine/cosine terms + 1 constant term for each of x, y, z)
MAX_ORDER = 10
MAX_DOFS = 3 * (2 * MAX_ORDER + 1)
MODEL_PATH = "saved_model.pt"
DATA_PATH = "saved_data.npz"
STATS_PATH = "scaling_stats.npz"


# Create a Model Class that inherits nn.Module. This neural network predicts coil length and mean squared curvature. 
class Model(nn.Module):
  def __init__(self):
    # Input layer, 3 hidden layers, output layer
    super().__init__() 
    self.fc1 = nn.Linear(MAX_DOFS, 512) 
    self.fc2 = nn.Linear(512, 512)     
    self.fc3 = nn.Linear(512, 512)
    self.fc4 = nn.Linear(512, 256)
    self.out = nn.Linear(256, 2)
    # TODO: Experiment with different architectures, activation functions, # of hidden layers, etc.

  def forward(self, x):
    # Forward pass through the network with ReLU activations
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = F.relu(self.fc4(x))
    x = self.out(x)
    return x

def generate_data(samples=100000, training=False):
    inputs, labels = [], []
    for i in range(samples):
        order = random.randint(1, 10)

        # Degrees of freedom: For each (x, y, z) coordinate there are 2 * order cosine/sine terms and one constant term. 
        num_dofs = 3 * (2 * order + 1)

        initial_coefficients = []
        for coord in range(3):
            # Constant term
            initial_coefficients.append((np.random.rand() - 0.5) * 2.0 * 2.0)
            for n in range(1, order+1):
                    # Harmonic scaling to prioritize earlier coefficients
                    scale = 2.0 / n
                     # Sine term
                    initial_coefficients.append((np.random.rand() - 0.5) * 2.0 * scale)
                    # Cosine term
                    initial_coefficients.append((np.random.rand() - 0.5) * 2.0 * scale)

        # Pad end of vector with zeroes to ensure consistent input size
        initial_coefficients = np.array(initial_coefficients)
        coefficients = np.pad(initial_coefficients, (0, MAX_DOFS - num_dofs), 'constant', constant_values=(0, 0))

        # Generate coil
        curve = CurveXYZFourier(quadpoints=100, order=order)
        curve.x = initial_coefficients

        # Calculate coil's total length
        length = CurveLength(curve).J()
        
        #Calculate mean squared curvature
        msc = MeanSquaredCurvature(curve).J()

        # Add coefficients and length to inputs and labels
        inputs.append(coefficients)
        labels.append((float(length), float(msc)))

    # Save data to .npz file, so we don't need to generate new data every training cycle.
    # Converting to numpy arrays for easy compression
    inputs_array = np.array(inputs, dtype=np.float32)
    labels_array = np.array(labels, dtype=np.float32) 
    scaled_inputs_array, scaled_labels_array = scale_data(inputs_array, labels_array)
    if training:
        np.savez(DATA_PATH, inputs=scaled_inputs_array, labels=scaled_labels_array) 
    return inputs_array, scaled_labels_array
def scale_data(inputs, labels):
    """ Since length will always be much larger than mean squared curvature, the optimizer would heavily prioritize 
        minimizing the length's loss. This is because length's loss function would dominate msc's loss function. 
        To ensure the model prioritizes both equally, we will scale length and msc to their respective z-scores and train 
        the model. When testing, we can unscale to generate true prediction. We also remove outliers, as garbage coils 
        can have msc values much larger (often over 1000x) than realistic coils."""
    
    mean_len = np.mean(labels[:, 0])
    std_len = np.std(labels[:, 0])

    msc_values = labels[:, 1]
    upper = np.percentile(msc_values, 95)
    clipped_msc = np.clip(msc_values, 0, upper)

    mean_msc = np.mean(clipped_msc)
    std_msc = np.std(clipped_msc)

    scaled_labels = labels.copy()
    scaled_labels[:, 0] = (labels[:, 0] - mean_len) / std_len
    scaled_labels[:, 1] = (clipped_msc - mean_msc) / std_msc

    mean_inputs = np.mean(inputs, axis=0)
    std_inputs = np.std(inputs, axis=0)
    std_inputs[std_inputs == 0] = 1.0
    scaled_inputs = (inputs - mean_inputs) / std_inputs
    np.savez(STATS_PATH, mean_len=mean_len, std_len=std_len, mean_msc=mean_msc, std_msc=std_msc, mean_inputs = mean_inputs, std_inputs = std_inputs)
    return scaled_inputs, scaled_labels

def train_model(inputs, labels):
   # Load model, split data into training and testing sets
   model = Model()

   X = torch.tensor(np.array(inputs), dtype=torch.float32)
   Y = torch.tensor(np.array(labels), dtype=torch.float32)

   # Wraps tensors into a dataset that can be indexed and iterated over
   dataset = TensorDataset(X, Y)

   # Load data into iterable mini-batches with shuffling
   loader = DataLoader(dataset, batch_size=16384, shuffle=True)

   # Define loss function and optimizer
   criterion = nn.MSELoss()
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

   for epoch in range(10):
        model.train() # Set model to training mode
        epoch_loss = 0
        epoch_loss_len = 0
        epoch_loss_msc = 0
        for batch_x, batch_y in loader:
            # Forward pass
            Y_Pred = model.forward(batch_x)
            loss_len = criterion(Y_Pred[:, 0], batch_y[:, 0])
            loss_msc = criterion(Y_Pred[:, 1], batch_y[:, 1])
            loss = loss_len + (20.0 * loss_msc) # Applying a scalar multiplier to the harder task (predicting curvature)         
            
            # Backward pass
            optimizer.zero_grad()
            loss_len.backward(retain_graph=True)  # Compute gradients for length
            loss_msc.backward()  # Compute gradients for MSC
            optimizer.step()

            # Accumulate loss for reporting
            epoch_loss += loss
            epoch_loss_len += loss_len
            epoch_loss_msc += loss_msc   
        avg_loss = epoch_loss / len(loader)
        avg_loss_len = epoch_loss_len / len(loader)
        avg_loss_msc = epoch_loss_msc / len(loader)
        if not epoch:
            print("Losses are in scaled units (stds). Loss of 1.0 means the prediction is off by 1 standard deviation on average.")
        print(f'Epoch {epoch+1}, Loss: {avg_loss}, Length Loss: {avg_loss_len}, MSC Loss: {avg_loss_msc}')
   # Save the trained model
   torch.save(model.state_dict(), MODEL_PATH)

if __name__ == "__main__":
    # Check for valid command-line arguments to either load data from a .npz file or generate new data
    if sys.argv.__len__() > 2 or sys.argv.__len__() < 1:
        print("Usage: python train.py [data_samples.npz]")
        sys.exit(1)
    elif sys.argv.__len__() == 2:
        datafile = sys.argv[1]
        cwd = os.getcwd()
        files = os.listdir(cwd)
        if datafile not in files or not datafile.endswith('.npz'):
            print("Please provide a valid .npz data file.")
            sys.exit(1)
        print(f"Fetching data from {datafile}...")
        data = np.load(datafile)
        inputs = data["inputs"]
        labels = data["labels"]
    else:
        print("Generating data...")
        inputs, labels = generate_data(5000000, True)

    print("Training model...")
    train_model(inputs, labels)

    print(f'Model saved to: {MODEL_PATH}')