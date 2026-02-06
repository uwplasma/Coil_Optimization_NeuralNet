import torch
import torch.nn as nn
import train
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os



def get_predictions(model, inputs):
    model.eval() # Set model to evaluation mode 

    # Disable gradient computation/tracking to speed up inference
    with torch.no_grad():
        X = torch.tensor(np.array(inputs), dtype=torch.float32)
        predictions = model(X)
        lengths = np.array(predictions[:, 0].cpu())
        mscs = np.array(predictions[:, 1].cpu())
        predictions = np.column_stack((lengths, mscs))
    return predictions


def evaluate_accuracy(labels, predictions):
    # Calculate MSE on unscaled data
    mse_length = np.mean((predictions[:, 0] - labels[:, 0])**2)
    mse_msc = np.mean((predictions[:, 1] - labels[:, 1])**2)
   
    # Calculate MAPE (mean absolute percentage error)
    length_mape = np.mean(np.abs((predictions[:, 0] - labels[:, 0]) / labels[:, 0])) * 100
    msc_mape = np.mean(np.abs((predictions[:, 1] - labels[:, 1]) / labels[:, 1])) * 100
    
    # Calculate "accuracy" as 100 - MAPE
    length_accuracy = 100 - length_mape
    msc_accuracy = 100 - msc_mape

    print(f'Average test loss length: {mse_length:.2f}')
    print(f'Average test loss msc: {mse_msc:.2f}')
    print(f'Average length accuracy: {length_accuracy:.2f}%')
    print(f'Average msc accuracy: {msc_accuracy:.2f}%\n')


"""
Evaluate accuracy by length bins (0-10, 10-20, 20-30, etc.) to see if model performs better on certain length ranges.
"""
def evaluate_accuracies_by_length(actual, predictions):
    preds = predictions[:, 0]
    labels = actual[:, 0]
    accuracies = []

    max_length = np.max(labels)
    iterations = int(max_length // 10 + 1)
    print(f"Total bins: {iterations}, Max length: {max_length}")
    for i in range(iterations):
        # Filter samples where length falls within the current bin (i*10 to (i+1)*10)
        filter = labels // 10 == i
        filtered_labels = labels[filter]
        filtered_preds = preds[filter]

        if len(filtered_labels) == 0:
           accuracies.append(np.nan)
           continue 

        # Calculate accuracy as 100 - MAPE for the current bin
        accuracy = 100 - np.mean(abs((filtered_preds - filtered_labels) / filtered_labels)) * 100
        accuracies.append(accuracy)
    
    # Print accuracies for each bin
    for i in range(len(accuracies)):
       print(f'Accuracy of coils size {i*10}-{i*10+10} = {accuracies[i]:.2f}%')

if __name__ == "__main__":
    # Load saved parameters into new model instance
    model = train.Model()
    print("Loading model...")
    if not os.path.exists(train.MODEL_PATH):
        print("No saved model found! Please train a model before testing.")
        exit()
    checkpoint = torch.load(train.MODEL_PATH)
    model.load_state_dict(checkpoint)

    print("Generating testing data...")
    inputs, labels = train.generate_data()

    print("Generating predictions...")
    predictions = get_predictions(model, inputs)

    # Scale back to original units
    stats = np.load(train.STATS_PATH)
    std_len = stats['std_len'].item()
    mean_len = stats['mean_len'].item()
    std_msc = stats['std_msc'].item()
    mean_msc = stats['mean_msc'].item()
    predictions[:, 0] = predictions[:, 0] * std_len + mean_len
    predictions[:, 1] = predictions[:, 1] * std_msc + mean_msc
    labels[:, 0] = labels[:, 0] * std_len + mean_len
    labels[:, 1] = labels[:, 1] * std_msc + mean_msc

    # Evaluate overall accuracy
    print("Evaluating model accuracy...")
    evaluate_accuracy(labels, predictions)

    print("Evaluating model accuracy by length bin (0-10, 10-20, 20-30, etc.)...")
    evaluate_accuracies_by_length(labels, predictions)