import sctp, socket
import time, random
from datetime import datetime

# PyTorch and Neural Network Imports
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

# Logging utility
from log import *

CONFIDENCE_THRESHOLD = 0.5
SAMPLING_RATE = 15360000

spectrogram_time = 0.005  # 5 ms
num_of_samples = int(SAMPLING_RATE * spectrogram_time)
spectrogram_size = num_of_samples * 8  # size in bytes, where 8 bytes is the size of one sample (complex64)

cmds = {
    'BASE_STATION_ON': b'y',
    'BASE_STATION_OFF': b'n',
    'ENABLE_ADAPTIVE_MCS': b'm',
    'DISABLE_ADAPTIVE_MCS': b'z',
}

current_iq_data = None
server = None

# Define the Neural Network Model
class SimpleModel(nn.Module):
    def __init__(self, num_bits_per_symbol):
        super(SimpleModel, self).__init__()
        scale = 8
        self.linear1 = nn.Linear(in_features=2, out_features=scale * num_bits_per_symbol)
        self.linear2 = nn.Linear(in_features=scale * num_bits_per_symbol, out_features=num_bits_per_symbol)
        self.activation = nn.ReLU()

    def forward(self, inputs):
        y = inputs
        z = torch.stack([y.real, y.imag], dim=0).permute(1, 2, 3, 0)  # Reshape for processing
        z = self.linear1(z)
        z = self.activation(z)
        z = self.linear2(z)
        z = torch.sigmoid(z)
        return z.flatten(-2, -1)

# Preprocess I/Q data for the model
def preprocess_iq_data(iq_data):
    complex_data = np.frombuffer(iq_data, dtype=np.complex64).reshape(64, -1)
    return torch.from_numpy(complex_data).to(torch.cfloat)

# Model prediction
def model_predict(model, iq_data):
    model.eval()
    inputs = preprocess_iq_data(iq_data).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(inputs)
    binary_predictions = (outputs > CONFIDENCE_THRESHOLD).float()
    return binary_predictions

# SCTP Framework

def init_e2(self):
    global server
    ip_addr = socket.gethostbyname(socket.gethostname())
    port = 5000
    log_info(self, f"E2-like enabled, connecting using SCTP on {ip_addr}")
    server = sctp.sctpsocket_tcp(socket.AF_INET)
    server.bind((ip_addr, port))
    server.listen()
    log_info(self, 'Server started')

def entry(self):
    global current_iq_data, server
    model = SimpleModel(num_bits_per_symbol=4)  # Initialize the neural network model
    model.load_state_dict(torch.load("simple_model.pth")['model_state_dict'])
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    init_e2(self)

    while True:
        try:
            conn, addr = server.accept()
            log_info(self, f'Connected by {addr}')

            while True:
                conn.send(f"E2-like request at {datetime.now().strftime('%H:%M:%S')}".encode('utf-8'))
                data = conn.recv(16384)
                while len(data) < spectrogram_size:
                    data += conn.recv(16384)
                
                log_info(self, f"Received buffer size {len(data)}")
                current_iq_data = data

                binary_predictions = model_predict(model, current_iq_data)
                log_info(self, f"Predicted Symbols: {binary_predictions}")

                result = "Interference" if torch.sum(binary_predictions) > 100 else "5G"
                if result == 'Interference':
                    conn.send(cmds['ENABLE_ADAPTIVE_MCS'])
                else:
                    conn.send(cmds['DISABLE_ADAPTIVE_MCS'])

        except OSError as e:
            log_error(self, e)

def start(thread=False):
    entry(None)

if __name__ == '__main__':
    start()
