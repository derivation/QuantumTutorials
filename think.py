import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import BasicAer, IBMQ, execute
from qiskit_aqua import QuantumInstance
from qiskit_aqua.algorithms import QSVMKernel
from qiskit_aqua.components.feature_maps import SecondOrderExpansion
from qiskit.tools.visualization import plot_histogram
from qsvm_datasets import *
from qiskit_aqua.utils import split_dataset_to_data_and_labels

alice = {'0': 854, '1': 146}

style = {'usepiformat': True, 'fontsize': 28, 'subfontsize': 22,
        'displaycolor': {'h': '#ffca64', 'u1': '#79acc7', 'ry': '#7aa3d3',
                        'z': '#d67272', 'x': '#86a790',
                        'target': '#ffffff', 'meas': '#cc95b5',
                        'cx': '#ffffff'}}

def read(number_of_features, training_size, test_size, gap):
    
    data = ad_hoc_data(training_size=training_size, test_size=test_size, n=number_of_features, gap=gap, PLOT_DATA=True)
    
    sample_Total, training_input, test_input, class_labels = data
    
    datapoints, class_to_label = split_dataset_to_data_and_labels(test_input)
    
    return training_input, test_input, datapoints, sample_Total


def plot_results(my_data, result):

    test_size = int(len(result['predicted_labels']) / 2)

    right_results = []
    wrong_results = []

    for x in range(test_size):
        if result['predicted_labels'][x] != 0:
            wrong_results.append(my_data[1]['A'][x])
        elif result['predicted_labels'][x] == 0:
            right_results.append(my_data[1]['A'][x])

    for x in range(test_size):
        if result['predicted_labels'][x + test_size] != 1:
            wrong_results.append(my_data[1]['B'][x])
        elif result['predicted_labels'][x + test_size] == 1:
            right_results.append(my_data[1]['B'][x])

    f, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.imshow(np.asmatrix(my_data[3]).T, interpolation='nearest', origin='lower', cmap='PiYG',
              extent=[0, 2 * np.pi, 0, 2 * np.pi])

    colors = ['#5aecad', '#f161fc', '#ff8100']
    ax.scatter(my_data[1]['A'][:, 0], my_data[1]['A'][:, 1], c=colors[0], s=100)
    ax.scatter(my_data[1]['B'][:, 0], my_data[1]['B'][:, 1], c=colors[1], s=100)

    if not wrong_results:
        pass
    else:
        for i in range(len(wrong_results)):
            ax.scatter(wrong_results[i][0], wrong_results[i][1], marker='X', s=400, c=colors[2])

    ax.set_xlim((0, 2 * np.pi))
    ax.set_ylim((0, 2 * np.pi))

    f.show()



