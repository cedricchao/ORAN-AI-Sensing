# from deeplearning.AIsim_main2 import *
# # from AIsim_main2 import OFDMModulator, OFDMDemodulator, MyLMMSEEqualizer, sim_ber, ber_plot


# # buffer size is the size threshold to trigger the process to make sure there is enough data
# # this is temporary
# buffer_size = 16384 * N  # Define N based on desired frame length
# # current_iq_data = b

# # while True:
#     # data = conn.recv(16384)
#     # if data:
#     #     current_iq_data += data
#     #     if len(current_iq_data) >= buffer_size:
#     #         # Pass the frame to the processing pipeline
#     #         process_frame(self, current_iq_data)
#     #         current_iq_data = b  # Reset the buffer for the next frame
# def init_sctp_connection():
#     # Initialize the SCTP connection
#     ip_addr = socket.gethostbyname(socket.gethostname())
#     port = 5000

#     # Set up the SCTP server
#     server = sctp.sctpsocket_tcp(socket.AF_INET)
#     server.bind((ip_addr, port))
#     server.listen()
    
#     print(f"SCTP server started at {ip_addr}:{port}")
#     return server

    

# def process_frame(self, iq_data):
#     # Encapsulate the entire processing pipeline for each frame
#     # Step 1: Modulate to frequency domain
#     modulated_data = OFDMModulator(iq_data)
    
#     # Step 2: Demodulate back to time domain
#     demodulated_data = OFDMDemodulator(modulated_data)

#     # Pass the processed data to the next stage
#     perform_channel_estimation(self, demodulated_data)



# def perform_channel_estimation(self, demodulated_data):
#     # Get channel estimates
#     channel_estimates = MyLMMSEEqualizer(demodulated_data)
#     snr = calculate_snr(channel_estimates)  # Custom function to derive SNR

#     # Decide RAN control based on channel quality
#     if snr < SNR_THRESHOLD:
#         conn.send(cmds['ENABLE_ADAPTIVE_MCS'])
#     else:
#         conn.send(cmds['DISABLE_ADAPTIVE_MCS'])

#     # Pass data to error rate and neural network
#     calculate_error_metrics(self, demodulated_data)




# def calculate_error_metrics(self, data):
#     ber = sim_ber(data)
#     if ber > BER_THRESHOLD:
#         conn.send(cmds['REDUCE_POWER'])
#     else:
#         conn.send(cmds['INCREASE_POWER'])

#     # Optionally log or visualize error metrics
#     ber_plot(ber)


# def run_interference_detection(self, data):
#     prediction = AIsensing_model.predict(data)
#     if prediction == 'Interference':
#         conn.send(cmds['ENABLE_ADAPTIVE_MCS'])
#     else:
#         conn.send(cmds['DISABLE_ADAPTIVE_MCS'])

# def main():
#     global current_iq_data
    
#     # Initialize SCTP connection
#     server = init_sctp_connection()

#     while True:
#         try:
#             # Accept an incoming connection from nodeB
#             conn, addr = server.accept()
#             print(f"Connected by {addr}")

#             while True:
#                 # Receive data in chunks of 16384 bytes
#                 data = conn.recv(16384)
#                 if data:
#                     current_iq_data += data
#                     if len(current_iq_data) >= buffer_size:
#                         # Pass the frame to the processing pipeline
#                         process_frame(current_iq_data)
#                         current_iq_data = b  # Reset the buffer for the next frame

#         except OSError as e:
#             print(f"Connection error: {e}")
#             continue

# # Run the main function
# if __name__ == "__main__":
#     main()

import numpy as np
import torch
import torch.nn as nn
import socket
import sctp
from datetime import datetime
import time


# === Symbol Detection Components === #

def BinarySource(shape):
    return np.random.randint(2, size=shape).astype(np.float32)


def complex_normal(shape, var=1.0):
    stddev = np.sqrt(var / 2)
    return np.random.normal(0.0, stddev, shape) + 1j * np.random.normal(0.0, stddev, shape)


def CreateConstellation(constellation_type, num_bits_per_symbol, normalize=True):
    if constellation_type != "qam":
        raise ValueError("Currently, only QAM constellations are supported.")

    # Generate QAM constellation
    c = np.zeros(2 ** num_bits_per_symbol, dtype=np.complex64)
    for i in range(2 ** num_bits_per_symbol):
        b = np.array(list(np.binary_repr(i, num_bits_per_symbol)), dtype=np.int16)
        c[i] = (1 - 2 * b[0]) + 1j * (1 - 2 * b[1])

    if normalize:
        c /= np.sqrt(np.mean(np.abs(c) ** 2))
    return c


class Mapper:
    def __init__(self, constellation_type="qam", num_bits_per_symbol=4):
        self.num_bits_per_symbol = num_bits_per_symbol
        self.points = CreateConstellation(constellation_type, num_bits_per_symbol)
        self.binary_base = 2 ** np.arange(num_bits_per_symbol - 1, -1, -1, dtype=int)

    def create_symbol(self, bits):
        bits_reshaped = bits.reshape(-1, self.num_bits_per_symbol)
        indices = np.dot(bits_reshaped, self.binary_base).astype(np.int32)
        return self.points[indices]


class SimpleModel(nn.Module):
    def __init__(self, num_bits_per_symbol):
        super(SimpleModel, self).__init__()
        scale = 8
        self.fc1 = nn.Linear(2, scale * num_bits_per_symbol)
        self.fc2 = nn.Linear(scale * num_bits_per_symbol, num_bits_per_symbol)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.cat([x.real.unsqueeze(-1), x.imag.unsqueeze(-1)], dim=-1)
        x = self.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x.flatten(-2, -1)


# === xApp Framework === #

class xApp:
    def __init__(self):
        self.server = None
        self.model = SimpleModel(num_bits_per_symbol=4)
        self.mapper = Mapper(constellation_type="qam", num_bits_per_symbol=4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.load_state_dict(torch.load("simple_model.pth")["model_state_dict"])
        self.model.eval()
        self.num_bits_per_symbol = 4

    def init_e2(self):
        ip_addr = socket.gethostbyname(socket.gethostname())
        port = 5000
        self.server = sctp.sctpsocket_tcp(socket.AF_INET)
        self.server.bind((ip_addr, port))
        self.server.listen()
        print(f"xApp listening on {ip_addr}:{port}")

    def preprocess_iq_data(self, iq_data):
        bits = BinarySource((64, 1024))
        symbols = self.mapper.create_symbol(bits)
        noise = complex_normal(symbols.shape, 1.0) * np.sqrt(0.1)  # Adjust noise scaling
        noisy_symbols = symbols + noise
        return torch.tensor(noisy_symbols).to(self.device).float(), bits

    def run_prediction(self, iq_data):
        samples, true_bits = self.preprocess_iq_data(iq_data)
        predictions = self.model(samples)
        predictions = torch.round(predictions).cpu().numpy()
        return predictions, true_bits

    def entry(self):
        self.init_e2()

        while True:
            try:
                conn, addr = self.server.accept()
                print(f"Connected by {addr}")

                while True:
                    iq_data = conn.recv(16384)
                    if iq_data:
                        predictions, true_bits = self.run_prediction(iq_data)

                        # Example decision logic
                        if predictions.mean() > 0.5:  # High error rate
                            conn.send(b"Interference detected, enabling adaptive MCS.")
                        else:
                            conn.send(b"No interference detected, disabling adaptive MCS.")
            except Exception as e:
                print(f"Error: {e}")
                continue


# === Main Entry Point === #
if __name__ == "__main__":
    xapp = xApp()
    xapp.entry()
