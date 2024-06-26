import numpy as np
import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()
from PySpice.Doc.ExampleTools import find_libraries
from PySpice.Probe.Plot import plot
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
import csv
import math
import pandas as pd
from numpy import argmax

data = pd.read_csv('train.csv').values
np.random.shuffle(data)

data_dev = data[:100].T
Y_dev = data_dev[-1].astype(int)
X_dev = data_dev[:-1]

data_train = data[100:].T
Y_train = data_train[-1].astype(int)
X_train = data_train[:-1]


def conversion(rw1, rw2, rb1):
    sum_w1_rows = np.sum(rw1, axis=1) + rb1.flatten()
    sum_w2_rows = np.sum(rw2, axis=1)

    w1 = sum_w1_rows[:, np.newaxis] / rw1
    b1 = sum_w1_rows[:, np.newaxis] / rb1
    w2 = sum_w2_rows[:, np.newaxis] / rw2

    return w1, b1, w2

def deconversion(w1, b1, w2):
    rw1 = np.zeros_like(w1)
    rb1 = np.zeros_like(b1)
    rw2 = np.zeros_like(w2)

    for i in range(len(w1)):
        for j in range(w1.shape[1]):
            rw1[i, j] = (1 / w1[i, j]) * (np.sum(w1[i, :]) + b1[i, 0])
        rb1[i, 0] = (1 / b1[i, 0]) * (np.sum(w1[i, :]) + b1[i, 0])
    
    for k in range(w2.shape[1]):
        rw2[:, k] = (1 / w2[:, k]) * (np.sum(w2[:, k]))

    return rw1, rb1, rw2

def init_params():
    rw1 = np.abs(np.random.normal(10000, 5000, size=(3, 5)))
    rb1 = np.abs(np.random.normal(10000, 5000, size=(3, 1)))
    rw2 = np.abs(np.random.normal(10000, 5000, size=(2, 3)))
    
    w1, b1, w2 = conversion(rw1, rw2, rb1)
    print(w1, b1, w2, rw1, rb1, rw2)
    return w1, b1, w2, rw1, rb1, rw2

def ReLU(z, drop):
    return np.maximum(0, z - drop)

def flatten_and_combine(rw1, rb1, rw2):
    return np.concatenate((rw1.flatten(), rb1.flatten())), rw2.flatten()

def build_circuit(input_voltages, combined_w1_b1, w2):
    circuit = Circuit('NN Circuit')

    bias = 1
    input_voltages = list(input_voltages) + [bias]

    for i, voltage in enumerate(input_voltages, start=1):
        circuit.V(i, f'NODE_{i}', circuit.gnd, f'DC {voltage}V')

    for i, resistance_value in enumerate(combined_w1_b1):
        node_from = f'NODE_{i // 5 + 1}'
        node_to = f'NODE_{chr(65 + (i % 5))}'
        circuit.R(f'R_{i+1}', node_from, node_to, f'{resistance_value:.2f}@Ω')

    circuit.model('Diode', 'D', IS=4.352E-9, N=1.906, BV=110, IBV=0.0001, RS=0.6458, CJO=7.048E-13,
                  VJ=0.869, M=0.03, FC=0.5, TT=3.48E-9)

    for i in range(3):
        node_label = chr(65 + i)
        circuit.Diode(name=f'D_{node_label}', anode=f'NODE_DIODE_{node_label}', cathode=f'NODE_{node_label}', model='Diode')
        circuit.R(f'R_DIODE_{node_label}', f'NODE_DIODE_{node_label}', f'NODE_{node_label}{node_label}', '4000@Ω')
        circuit.R(f'R_GND_{node_label}', f'NODE_{node_label}{node_label}', circuit.gnd, '4000@Ω')

    for i in range(3):
        node_label = chr(65 + i)
        for j in range(6):
            circuit.R(f'R_{node_label}_{j+1}', f'NODE_{node_label}{node_label}', f'NODE_VALUE_{j}', f'{w2[i]:.2f}@Ω')

    for j in range(2):
        circuit.R(f'R_VALUE_{j}', f'NODE_VALUE_{j}', circuit.gnd, '100000@Ω')

    return circuit

def forward_prop(w1, b1, w2, rw1, rb1, rw2, X, input_voltages):
    combined_w1_b1, w2_flatten = flatten_and_combine(rw1, rb1, rw2)
    new_circuit = build_circuit(input_voltages, combined_w1_b1, w2_flatten)
    simulator = new_circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.operating_point()

    node_voltages = []
    nodes_of_interest = ['NODE_A', 'NODE_AA', 'NODE_B', 'NODE_BB', 'NODE_C', 'NODE_CC']
    for node in nodes_of_interest:
        voltage = float(analysis.nodes[node.lower()])
        node_voltages.append(voltage)

    drop = np.array([
        node_voltages[0] - node_voltages[1],
        node_voltages[2] - node_voltages[3],
        node_voltages[4] - node_voltages[5]
    ])

    z1 = w1.dot(X) + b1
    a1 = np.zeros_like(z1)
    for i in range(z1.shape[0]):
        a1[i] = ReLU(z1[i], drop[i])

    z2 = w2.dot(a1)
    a2 = z2
    return z1, a1, z2, a2, drop

def one_hot(Y):
    one_hot_Y = np.zeros((Y.max() + 1, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

def deriv_ReLU(Z):
    return Z > 0

def back_prop(z1, a1, z2, a2, w2, X, Y, drop):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dz2 = a2 - one_hot_Y
    dw2 = 1 / m * dz2.dot(a1.T)
    dz1 = (w2.T.dot(dz2) * deriv_ReLU(z1)) + drop[:, np.newaxis]
    dw1 = 1 / m * dz1.dot(X.T)
    db1 = 1 / m * np.sum(dz1, axis=1, keepdims=True)
    return dw1, db1, dw2

def update_params(w1, b1, w2, dw1, db1, dw2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    rw1, rb1, rw2 = deconversion(w1, b1, w2)
    return w1, b1, w2, rw1, rb1, rw2

def get_predictions(z2):
    return argmax(z2, axis=0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha):
    w1, b1, w2, rw1, rb1, rw2  = init_params()
    for i in range(20):
        for j in range(X.shape[1]):
            input_voltages = X[:, j]
            z1, a1, z2, a2, drop = forward_prop(w1, b1, w2, rw1, rb1, rw2, X, input_voltages)
            dw1, db1, dw2 = back_prop(z1, a1, z2, a2, w2, X, Y, drop)
            w1, b1, w2, rw1, rb1, rw2 = update_params(w1, b1, w2, dw1, db1, dw2, alpha)

    if i % 10 == 0:
        predictions = get_predictions(z2)
        accuracy = get_accuracy(predictions, Y)
        print(f"Iteration {i}: Accuracy = {accuracy}")
        print(rw1, rw2, rb1)

    return w1, b1, w2

alpha = 0.015
print("-------------------------------------------------------------------------------------") 
print("alpha ", alpha)
w1, b1, w2 = gradient_descent(X_train, Y_train, alpha=alpha)