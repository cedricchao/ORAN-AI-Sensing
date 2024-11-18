from deeplearning.AIsim_main2 import *
# from AIsim_main2 import OFDMModulator, OFDMDemodulator, MyLMMSEEqualizer, sim_ber, ber_plot


# buffer size is the size threshold to trigger the process to make sure there is enough data
# this is temporary
buffer_size = 16384 * N  # Define N based on desired frame length
# current_iq_data = b

# while True:
    # data = conn.recv(16384)
    # if data:
    #     current_iq_data += data
    #     if len(current_iq_data) >= buffer_size:
    #         # Pass the frame to the processing pipeline
    #         process_frame(self, current_iq_data)
    #         current_iq_data = b  # Reset the buffer for the next frame
def init_sctp_connection():
    # Initialize the SCTP connection
    ip_addr = socket.gethostbyname(socket.gethostname())
    port = 5000

    # Set up the SCTP server
    server = sctp.sctpsocket_tcp(socket.AF_INET)
    server.bind((ip_addr, port))
    server.listen()
    
    print(f"SCTP server started at {ip_addr}:{port}")
    return server

    

def process_frame(self, iq_data):
    # Encapsulate the entire processing pipeline for each frame
    # Step 1: Modulate to frequency domain
    modulated_data = OFDMModulator(iq_data)
    
    # Step 2: Demodulate back to time domain
    demodulated_data = OFDMDemodulator(modulated_data)

    # Pass the processed data to the next stage
    perform_channel_estimation(self, demodulated_data)



def perform_channel_estimation(self, demodulated_data):
    # Get channel estimates
    channel_estimates = MyLMMSEEqualizer(demodulated_data)
    snr = calculate_snr(channel_estimates)  # Custom function to derive SNR

    # Decide RAN control based on channel quality
    if snr < SNR_THRESHOLD:
        conn.send(cmds['ENABLE_ADAPTIVE_MCS'])
    else:
        conn.send(cmds['DISABLE_ADAPTIVE_MCS'])

    # Pass data to error rate and neural network
    calculate_error_metrics(self, demodulated_data)




def calculate_error_metrics(self, data):
    ber = sim_ber(data)
    if ber > BER_THRESHOLD:
        conn.send(cmds['REDUCE_POWER'])
    else:
        conn.send(cmds['INCREASE_POWER'])

    # Optionally log or visualize error metrics
    ber_plot(ber)


def run_interference_detection(self, data):
    prediction = AIsensing_model.predict(data)
    if prediction == 'Interference':
        conn.send(cmds['ENABLE_ADAPTIVE_MCS'])
    else:
        conn.send(cmds['DISABLE_ADAPTIVE_MCS'])

def main():
    global current_iq_data
    
    # Initialize SCTP connection
    server = init_sctp_connection()

    while True:
        try:
            # Accept an incoming connection from nodeB
            conn, addr = server.accept()
            print(f"Connected by {addr}")

            while True:
                # Receive data in chunks of 16384 bytes
                data = conn.recv(16384)
                if data:
                    current_iq_data += data
                    if len(current_iq_data) >= buffer_size:
                        # Pass the frame to the processing pipeline
                        process_frame(current_iq_data)
                        current_iq_data = b  # Reset the buffer for the next frame

        except OSError as e:
            print(f"Connection error: {e}")
            continue

# Run the main function
if __name__ == "__main__":
    main()