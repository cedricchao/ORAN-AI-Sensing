
# Implementation and Setup of the AI Framework

## Setup Conda Environment in Linux

```bash
   curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
```

You can also install conda in silent mode, but you need to run additional commands to initialize PATH and perform init

```bash
   $ python3 -V #system's python3 version
   Python 3.10.12
   $ bash Miniconda3-latest-Linux-x86_64.sh -b -u
   $ source ~/miniconda3/bin/activate
   $ conda init bash
```
".bashrc" has been updated, and close and re-open your current shell to make changes effective.

## Setup Conda Environment in Mac OS
Download miniconda from [link](https://docs.anaconda.com/miniconda/)

```bash
% curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o Miniconda3-latest-MacOSX-x86_64.sh
% bash Miniconda3-latest-MacOSX-x86_64.sh -b -u
lkk@kaikais-mbp2019 Developers % source ~/miniconda3/bin/activate
(base) lkk@kaikais-mbp2019 Developers % conda init bash
#remove existing environment
conda remove -n ENV_NAME --all
% conda create --name py310 python=3.10
% conda activate py310
% conda install -c conda-forge pylibiio
% pip install pyadi-iio
% pip install numpy matplotlib
```

In Mac, if you face `md5sum: command not found` problem, install it via `brew`
```bash
brew update
brew upgrade
brew install md5sha1sum
```
## Conda Virtual Environment
Create a Conda virtual environment with python 3.10 (`tensorrt==8.5.3.1` does not support python3.11):

```bash
   $ conda info --envs #check existing conda environment
   $ conda create --name py310cu118 python=3.10
   $ conda activate py310cu118
   $ conda info
   $ conda deactivate #To deactivate an active environment
```

Windows Side: Install cuda, cudnn, tensorflow, and pytorch (Windows Native - Windows 7 or higher (64-bit) (no GPU support after TF 2.10))
```bash
   #install cuda under Conda
   conda install -y cuda -c nvidia/label/cuda-11.8.0 #new method from https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#conda-installation
   # Install pytorch
   conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   # install cuDNN and Tensorflow
   #python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.0
   #pip install nvidia-cudnn-cu11==8.7.0.84
   #pip install tensorflow[and-cuda]==2.14.0
   (py310cu118) PS D:\Developer\radarsensing> pip install tensorflow==2.14.0
   (py310cu118) PS D:\Developer\radarsensing> pip install nvidia-cudnn-cu11
   Successfully installed nvidia-cublas-cu11-11.11.3.6 nvidia-cuda-nvrtc-cu11-11.8.89 nvidia-cudnn-cu11-9.1.1.17
   python -c "import tensorflow as tf; print('tf version:', tf.__version__); print(tf.config.list_physical_devices('GPU'))"
   # Verify Pytorch installation
   python -c "import torch; print('Torch version:', torch.__version__); print(torch.cuda.is_available())"
```

In Linux:
```bash
(py310cu118) lkk@ThinkpadX1:~/MyRepo$ python3 -m pip install nvidia-cudnn-cu11==8.7.0.84
(py310cu118) lkk@ThinkpadX1:~/MyRepo$ pip install tensorflow[and-cuda]==2.14.0
Successfully installed absl-py-2.1.0 astunparse-1.6.3 cachetools-5.3.3 certifi-2024.7.4 charset-normalizer-3.3.2 flatbuffers-24.3.25 gast-0.6.0 google-auth-2.31.0 google-auth-oauthlib-1.0.0 google-pasta-0.2.0 grpcio-1.64.1 h5py-3.11.0 idna-3.7 keras-2.14.0 libclang-18.1.1 markdown-3.6 ml-dtypes-0.2.0 numpy-2.0.0 nvidia-cublas-cu11-11.11.3.6 nvidia-cuda-cupti-cu11-11.8.87 nvidia-cuda-nvcc-cu11-11.8.89 nvidia-cuda-runtime-cu11-11.8.89 nvidia-cudnn-cu11-8.7.0.84 nvidia-curand-cu11-10.3.0.86 nvidia-cusolver-cu11-11.4.1.48 nvidia-cusparse-cu11-11.7.5.86 nvidia-nccl-cu11-2.16.5 oauthlib-3.2.2 opt-einsum-3.3.0 packaging-24.1 pyasn1-0.6.0 pyasn1-modules-0.4.0 requests-2.32.3 requests-oauthlib-2.0.0 rsa-4.9 six-1.16.0 tensorboard-2.14.1 tensorboard-data-server-0.7.2 tensorflow-2.14.0 tensorflow-estimator-2.14.0 tensorflow-io-gcs-filesystem-0.37.1 tensorrt-8.5.3.1 termcolor-2.4.0 urllib3-2.2.2 wrapt-1.14.1
(py310cu118) lkk@ThinkpadX1:~/MyRepo$ mkdir -p $CONDA_PREFIX/etc/conda/activate.d
(py310cu118) lkk@ThinkpadX1:~/MyRepo$ echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
(py310cu118) lkk@ThinkpadX1:~/MyRepo$ echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
(py310cu118) lkk@ThinkpadX1:~/MyRepo$ source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
$ cat /home/lkk/miniconda3/envs/py310cu118/lib/python3.10/site-packages/nvidia/cudnn/include/cudnn_version.h
#define CUDNN_MAJOR 8
#define CUDNN_MINOR 7
(py310cu118) lkk@ThinkpadX1:~/MyRepo$ python3 -c "import tensorflow as tf; print('tf version:', tf.__version__); print(tf.config.list_physical_devices('GPU'))"
pip uninstall numpy
pip install numpy==1.26.4
```

In Windows WSL2:
```bash
(base) lkk@newalienware:~/Developer/AIsensing$ conda create --name py310cu118 python=3.10
(base) lkk@newalienware:~/Developer/AIsensing$ conda activate py310cu118
(py310cu118) lkk@newalienware:~/Developer/AIsensing$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
(py310cu118) lkk@newalienware:~/Developer/AIsensing$ pip install tensorflow[and-cuda]==2.14.0
(py310cu118) lkk@newalienware:~/Developer/AIsensing$ mkdir -p $CONDA_PREFIX/etc/conda/activate.d
(py310cu118) lkk@newalienware:~/Developer/AIsensing$ echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
(py310cu118) lkk@newalienware:~/Developer/AIsensing$ echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
(py310cu118) lkk@newalienware:~/Developer/AIsensing$ source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
(py310cu118) lkk@newalienware:~/Developer/AIsensing$ cat /home/lkk/miniconda3/envs/py310cu118/lib/python3.10/site-packages/nvidia/cudnn/include/cudnn_version.h
(py310cu118) lkk@newalienware:~/Developer/AIsensing$ pip uninstall numpy
  Successfully uninstalled numpy-2.1.2
(py310cu118) lkk@newalienware:~/Developer/AIsensing$ pip install numpy==1.26.4
(py310cu118) lkk@newalienware:~/Developer/AIsensing$ python3 -c "import tensorflow as tf; print('tf version:', tf.__version__); print(tf.config.list_physical_devices('GPU'))"
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
(py310cu118) lkk@newalienware:~/Developer/AIsensing$ conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Install other python packages:
```bash
   conda install -y -c conda-forge jupyterlab
   conda install -y ipykernel
   jupyter kernelspec list #view current jupyter kernels
   ipython kernel install --user --name=py310cu118
   conda install -y matplotlib pandas Pillow scipy pyyaml scikit-image 
   pip install pyqt5 pyqt6 PySide6 pyqtgraph opencv-python-headless PyOpenGL PyOpenGL_accelerate pyopengl
   pip install sionna DeepMIMO pyadi-iio
   pip install opencv-python --upgrade
   $ python ./sdrpysim/testmatplotlibcv2.py #test matplotlib and cv2
   $ pip install PyQt6
   $ sudo apt-get install libegl1-mesa libegl1-mesa-dev
   $ ldconfig -p | grep libEGL
   $ pip install pyqtgraph
   $ python ./sdrpysim/pyqt6qtgraphtest.py

   $ pip install sionna
   Successfully installed absl-py-2.1.0 asttokens-2.4.1 astunparse-1.6.3 decorator-5.1.1 drjit-0.4.6 executing-2.1.0 flatbuffers-24.3.25 gast-0.6.0 google-auth-2.35.0 google-auth-oauthlib-1.2.1 google-pasta-0.2.0 grpcio-1.67.0 h5py-3.12.1 importlib-resources-6.4.5 ipydatawidgets-4.3.2 ipython-8.28.0 ipywidgets-8.0.5 jedi-0.19.1 jupyterlab-widgets-3.0.5 keras-2.15.0 libclang-18.1.1 markdown-3.7 matplotlib-inline-0.1.7 mitsuba-3.5.2 ml-dtypes-0.3.2 oauthlib-3.2.2 opt-einsum-3.4.0 parso-0.8.4 pexpect-4.9.0 prompt-toolkit-3.0.48 protobuf-4.25.5 ptyprocess-0.7.0 pure-eval-0.2.3 pyasn1-0.6.1 pyasn1-modules-0.4.1 pythreejs-2.4.2 requests-oauthlib-2.0.0 rsa-4.9 sionna-0.19.0 stack-data-0.6.3 tensorboard-2.15.2 tensorboard-data-server-0.7.2 tensorflow-2.15.1 tensorflow-estimator-2.15.0 tensorflow-io-gcs-filesystem-0.37.1 termcolor-2.5.0 traitlets-5.14.3 traittypes-0.2.1 wcwidth-0.2.13 werkzeug-3.0.4 widgetsnbextension-4.0.13 wrapt-1.14.1
```

Installation in Mac
```bash
% pip install pyadi-iio
(mypy310) (base) kaikailiu@Kaikais-MBP radarsensing % pip install tensorflow==2.14.0
#Test tensorflow
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
pip install sionna
```

## Test NVIDIA Sionna

Install Nvidia sionna from [SionnaGithub](https://github.com/NVlabs/sionna/tree/main):
```bash
$ pip install sionna
(mycondapy310) (base) lkk@lkk-intel12:~/Developer$ git clone https://github.com/NVlabs/sionna.git
```

Nvidia sionna requires Tensorflow (2.10 - 2.14). Latest version of Tensorflow (e.g., 2.16) may not support due to the `complex` data type for tf layers. Sionna also requires Tensorflow with cuda backend. If cuda is not available, LLVM backend is required. If your tensorflow cannot detect the GPU and you do not have the LLVM backend, it may show the following error: `it_init_thread_state(): the LLVM backend is inactive because the LLVM shared library ("libLLVM.so") could not be found! Set the DRJIT_LIBLLVM_PATH environment variable to specify its path.`

Run `deeplearning/MIMO_OFDM_Transmissions_over_CDL.ipynb` to test the Nvidia sionna.

## DeepMIMO
[DeepMIMO](https://deepmimo.net/) is a generic dataset that enables a wide range of machine/deep learning applications for MIMO systems. It takes as input a set of parameters (such as antenna array configurations and time-domain/OFDM parameters) and generates MIMO channel realizations, corresponding locations, angles of arrival/departure, etc., based on these parameters and on a ray-tracing scenario selected from those available in DeepMIMO.

DeepMIMO provides multiple scenarios that one can select from. We use the O1 scenario with the carrier frequency set to 60 GHz (O1_60). We need to download the "O1_60" data files from this [page](https://deepmimo.net/scenarios/o1-scenario/). The downloaded zip file should be extracted into a folder, and the parameter DeepMIMO_params['dataset_folder'] should be set to point to this folder. To use DeepMIMO, the DeepMIMO dataset first needs to be generated. The generated DeepMIMO dataset contains channels for different locations of the users and basestations. The layout of the O1 scenario is shown in the figure below.
![DeepMIMO O1](../imgs/deepmimo_o1.png)

Install DeepMIMO python package
```bash
pip install DeepMIMO
```

In our `deepMIMO5.py` file, we need to setup the `dataset_folder='data'` in the main file, or setup `dataset_folder=r'D:\Dataset\CommunicationDataset\O1_60'` in Windows side. It will use the following function to get the DeepMIMO dataset:
```bash
DeepMIMO_dataset = get_deepMIMOdata(scenario=scenario, dataset_folder=dataset_folder, showfig=showfig)
```

We can load default parameters of `DeepMIMO` and setup our own parameters
```bash
# Load the default parameters
parameters = DeepMIMO.default_params()
parameters['num_paths'] = 10
parameters['active_BS'] = np.array([1]) # Basestation indices to be included in the dataset, total 12 bs in O1
#parameters['active_BS'] = np.array([1, 5, 8]) #enable multiple basestations
parameters['user_row_first'] = 1 #400 # First user row to be included in the dataset
parameters['user_row_last'] = 100 #450 # Last user row to be included in the dataset
```
The generated DeepMIMO dataset contains channels for different locations of the users and basestations. In our example, the users located on the rows `user_row_first` to `user_row_first`. Each of these rows consists of 181 user locations, resulting in `181*100=18100` basestation-user channels.

After setup all parameters, `DeepMIMO_dataset = DeepMIMO.generate_data(parameters)` will generate the `DeepMIMO_dataset`.

The antenna arrays in the DeepMIMO dataset are defined through the x-y-z axes. In the following example, a single-user MISO downlink is considered. The basestation is equipped with a uniform linear array of 16 elements spread along the x-axis. The users are each equipped with a single antenna. We can check the parameters of the `DeepMIMO_dataset`:

```bash
# Number of basestations
print(len(DeepMIMO_dataset)) #1
# Keys of a basestation dictionary
print(DeepMIMO_dataset[0].keys()) #['user', 'basestation', 'location']
# Keys of a channel
print(DeepMIMO_dataset[0]['user'].keys()) #['paths', 'LoS', 'location', 'distance', 'pathloss', 'channel']
# Shape of the channel matrix
print(DeepMIMO_dataset[active_bs_idx]['user']['channel'].shape) #(num_ue_locations=18100, 1, bs_antenna=16, strongest_path=10) 
# The channel matrix between basestation i=0 and user j=0, Shape of BS 0 - UE 0 channel
print(DeepMIMO_dataset[active_bs_idx]['user']['channel'][j].shape) #(1, 16, 10)
```

Ray-tracing Path Parameters are saved in dictionary, number of path is 9, and each key is a size of 9 array.
```bash
# Path properties of BS 0 - UE 0
print(DeepMIMO_dataset[active_bs_idx]['user']['paths'][j]) #Ray-tracing Path Parameters in dictionary
#'num_paths': 9, Azimuth and zenith angle-of-arrivals – degrees (DoA_phi, DoA_theta), size of 9 array
# Azimuth and zenith angle-of-departure – degrees (DoD_phi, DoD_theta)
# Time of arrival – seconds (ToA)
# Phase – degrees (phase)
# Power – watts (power)
# Number of paths (num_paths)
print(DeepMIMO_dataset[active_bs_idx]['user']['LoS'][j]) #Integer of values {-1, 0, 1} indicates the existence of the LOS path in the channel.
# (1): The LoS path exists.
# (0): Only NLoS paths exist. The LoS path is blocked (LoS blockage).
# (-1): No paths exist between the transmitter and the receiver (Full blockage).

print(DeepMIMO_dataset[active_bs_idx]['user']['distance'][j])
#The Euclidian distance between the RX and TX locations in meters.

print(DeepMIMO_dataset[active_bs_idx]['user']['pathloss'][j])
#The combined path-loss of the channel between the RX and TX in dB.
```
The BS location and UE locations are shown in this figure. The first row has 181 user locations, total user locations are 18100
![UserStation Location](../imgs/ddeepmimo_userstationlocation.png)

The channel response is `(1, 16, 10)`, means 16 bs antenna, and 10 path components:
![Channel Response](../imgs/deepmimo_channelresponse.png)

The following two figures show the UE and BS path loss and positions:
![UE BS positions with path loss](../imgs/deepmimo-uebspositions.png)

![UEGrid Path Loss](../imgs/deepmimo_uegridpathloss.png)

In NVIDIA Sionna, `DeepMIMOSionnaAdapter(DeepMIMO_dataset, bs_idx, ue_idx)` (implemented in [sionna_adapter](https://github.com/DeepMIMO/DeepMIMO-python/blob/master/src/DeepMIMOv3/sionna_adapter.py)) is used to wrap the `DeepMIMO_dataset` and generate dataset. In our project, we did not use `DeepMIMOSionnaAdapter`, instead we create a pytorch dataset class `class DeepMIMODataset(Dataset)` to wrap the deepmimo dataset, each iteration get `h` and `tau`. 
```bash
self.channeldataset = DeepMIMODataset(DeepMIMO_dataset=DeepMIMO_dataset, ue_idx=ue_idx)
h, tau = next(iter(self.channeldataset)) #h: (1, 1, 1, 16, 10, 1), tau:(1, 1, 10)
#complex gains `h` and delays `tau` for each path
#print(h.shape) #[num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
#print(tau.shape) #[num_rx, num_tx, num_paths]
```
There are some additional settings in `DeepMIMOSionnaAdapter`
```bash
num_time_steps = 1 # Time step = 1 for static scenarios
# Determine the number of samples based on the given indices
self.num_samples_bs = self.bs_idx.shape[0]
self.num_samples_ue = self.ue_idx.shape[0]
self.num_samples = self.num_samples_bs * self.num_samples_ue

# Determine the number of tx and rx elements in each channel sample based on the given indices
self.num_rx = self.ue_idx.shape[1]
self.num_tx = self.bs_idx.shape[1]
# The required path power shape for Sionna
self.ch_shape = (self.num_rx, 
                  self.num_rx_ant, 
                  self.num_tx, 
                  self.num_tx_ant, 
                  self.num_paths, 
                  self.num_time_steps)

# The required path delay shape for Sionna
self.t_shape = (self.num_rx, self.num_tx, self.num_paths)
```
bs_idx can be set to a 2D numpy matrix of shape `# of samples * # of basestations per sample`, ue_idx can be set to a 2D numpy matrix of shape `# of samples * # of users per sample`. 

Similiar to the `CIRDataset class` in NVIDIA Sionna that used Tensorflow dataset, we create a pytorch `DataLoader` and define `batch_size`, so that each iteration we get a batch of `h` and `tau`:
```bash
self.data_loader = DataLoader(dataset=self.channeldataset, batch_size=batch_size, shuffle=True, pin_memory=True)
h_b, tau_b = next(iter(self.data_loader)) #h_b: [64, 1, 1, 1, 16, 10, 1], tau_b=[64, 1, 1, 10]
#print(h_b.shape) #[batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
#print(tau_b.shape) #[batch, num_rx, num_tx, num_paths]
tau_b=tau_b.numpy()#torch tensor to numpy
h_b=h_b.numpy()
```

The plot of the channel impulse response is shown here (max 10 paths)
![Channel Impulse Response](../imgs/deepmimo_chimpulse.png)

## Deep Learning with DeepMIMO Dataset
Install Pytorch and TensorFlow (some package needs Tensorflow). Following [Tensorflow Pip](https://www.tensorflow.org/install/pip) page to install Tensorflow:
```bash
(mypy310) lkk@Alienware-LKKi7G8:~/Developer/AIsensing$ python3 -m pip install tensorflow[and-cuda]
# Verify the installation:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Download [DeepMIMO](https://www.deepmimo.net/) dataset.

Follow [link](https://www.deepmimo.net/versions/v2-python/), install DeepMIMO python package:
```bash
pip install DeepMIMO
```

Select and download a scenario from the scenarios [page](https://www.deepmimo.net/scenarios/), for example, select Outdoor scenario1 (O1). Download 'O1_60' and 'O1_3p5' to the 'data' folder.

Run the DeepMIMO simulation and obtain the BER curve for various configurations:
```bash
python deeplearning/deepMIMO5_sim.py
```
[BER Curve](imgs/berlist.jpg)

## Get CIR from Other Channels
The $h_b$, $tau_b$ generated is Channel Impulse Response (CIR), $h_b$'s shape meaning is `complex [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]`, $tau_b$'s shape `float [batch, num_rx, num_tx, num_paths]`. We can also use simulation to generate CIR based on assumptions of Gaussian distributed i.i.d. path coefficients and uniformly distributed i.i.d. path delays:
```bash
# Random path coefficients
h_shape = [dataset_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
h = (np.random.normal(size=h_shape) + 1j*np.random.normal(size=h_shape))/np.sqrt(2)
# Random path delays
tau = np.random.uniform(size=[dataset_size, num_rx, num_tx, num_paths])
```

The instance `cdl` of the [CDL](https://nvlabs.github.io/sionna/api/channel.html#clustered-delay-line-cdl) [ChannelModel](https://nvlabs.github.io/sionna/api/channel.html#channel-model-interface) can be used to generate batches of random realizations of continuous-time
channel impulse responses, consisting of complex gains `a` and delays `tau` for each path. 
To account for time-varying channels, a channel impulse responses is sampled at the `sampling_frequency` for `num_time_samples` samples.
For more details on this, please have a look at the [API documentation](https://nvlabs.github.io/sionna/api/channel.html) of the channel models. In order to model the channel in the frequency domain, we need `num_ofdm_symbols` samples that are taken once per `ofdm_symbol_duration`, which corresponds to the length of an OFDM symbol plus the cyclic prefix.

```bash
a, tau = cdl(batch_size=32, num_time_steps=rg.num_ofdm_symbols, sampling_frequency=1/rg.ofdm_symbol_duration)
```

The path gains `a` have shape\
`[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]`\
and the delays `tau` have shape\
`[batch_size, num_rx, num_tx, num_paths]`.

If using CIR dataset, the `channel_model` will also generate $h$ and $tau$:
```bash
batch_size = 64 # The batch_size cannot be changed after the creation of the channel model
channel_model = channel.CIRDataset(generator,
                                      batch_size,
                                      num_rx,
                                      num_rx_ant,
                                      num_tx,
                                      num_tx_ant,
                                      num_paths,
                                      num_time_steps)
h, tau = channel_model()
```
It can also be generated by CDL channel model:
```bash
a, tau = cdl(batch_size=32, num_time_steps=rg.num_ofdm_symbols, sampling_frequency=1/rg.ofdm_symbol_duration)
```
The path gains a have shape `[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]` and the delays tau have shape `[batch_size, num_rx, num_tx, num_paths]`. The delays are assumed to be static within the time-window of interest. Only the complex path gains change over time. The last dimenstion `num_time_steps` means Time evolution of path gain.

### channel frequency responses (frequency-domain)
If we want to use the continuous-time channel impulse response to simulate OFDM transmissions under ideal conditions, i.e., no inter-symbol interference, inter-carrier interference, etc., we need to convert it to the frequency domain. For the simulation of communication system based on OFDM, we can use the channel model to generate channel frequency responses $h_{freq}$. 
```bash
ofdm_channel = channel.GenerateOFDMChannel(channel_model, resource_grid)
# Generate a batch of frequency responses
# Shape: [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]
h_freq = ofdm_channel()
```
The format of `h_freq` is `[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]`, where `num_ofdm_symbols` refers to the number of individual OFDM symbols within a transmission burst, e.g., `num_ofdm_symbols=14`, `num_subcarriers=76`. Each OFDM symbol contains multiple subcarriers (frequency bins). These symbols are grouped together to form a burst of data. 

This can also be done with the function `cir_to_ofdm_channel` that computes the Fourier transform of the continuous-time channel impulse response at a set of frequencies, corresponding to the different subcarriers. The frequencies can be obtained with the help of the convenience function subcarrier_frequencies.
```bash
frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)
#We can apply the channel frequency response to a given input
# Function that will apply the channel frequency response to an input signal
channel_freq = ApplyOFDMChannel(add_awgn=True)
```

In the member function `def generateChannel(self, x_rg, no, channeltype='ofdm'):` of `class Transmitter` in `deepMIMO5.py`, it contains the `cir_to_ofdm_channel` function and generate the channel frequency responses
```bash
# Generate the OFDM channel response
#computes the Fourier transform of the continuous-time channel impulse response at a set of `frequencies`, corresponding to the different subcarriers.
#h: [64, 1, 1, 1, 16, 10, 1], tau: [64, 1, 1, 10] => (64, 1, 1, 1, 16, 1, 76) 
h_freq = mygenerate_OFDMchannel(h_b, tau_b, self.fft_size, subcarrier_spacing=60000.0, dtype=np.complex64, normalize_channel=True)
#h_freq : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size]
# Generate the OFDM channel
channel_freq = MyApplyOFDMChannel(add_awgn=True)
#h_freq : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size]
#(64, 1, 1, 1, 16, 1, 76)
y = channel_freq([x_rg, h_freq, no]) #h_freq is array
#Channel outputs y : [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], complex    
#print(y.shape) #[64, 1, 1, 14, 76] dim (3,4 removed)
```

The figure of the channel frequency response is shown here:
![Channel Frequency Response](../imgs/ofdmchannelfreq.png)


### discrete-time impulse response (time-domain)
In the same way as we have created the frequency channel impulse response from the continuous-time response, we can use the latter to compute a discrete-time impulse response. This can then be used to model the channel in the time-domain through discrete convolution with an input signal. Time-domain channel modeling is necessary whenever we want to deviate from the perfect OFDM scenario, e.g., OFDM without cyclic prefix, inter-subcarrier interference due to carrier-frequency offsets, phase noise, or very high Doppler spread scenarios, as well as other single or multicarrier waveforms (OTFS, FBMC, UFMC, etc).

A discrete-time impulse response can be obtained with the help of the function cir_to_time_channel that requires a `bandwidth` parameter. This function first applies a perfect low-pass filter of the provided bandwith to the continuous-time channel impulse response and then samples the filtered response at the Nyquist rate. The resulting discrete-time impulse response is then truncated to finite length, depending on the delay spread. l_min and l_max denote truncation boundaries and the resulting channel has l_tot=l_max-l_min+1 filter taps. A detailed mathematical description of this process is provided in the API documentation of the channel models. You can freely chose both parameters if you do not want to rely on the default values.

In order to model the channel in the domain, the continuous-time channel impulse response must be sampled at the Nyquist rate. We also need now `num_ofdm_symbols x (fft_size + cyclic_prefix_length) + l_tot-1` samples in contrast to `num_ofdm_symbols` samples for modeling in the frequency domain. This implies that the memory requirements of time-domain channel modeling is significantly higher. We therefore recommend to only use this feature if it is really necessary. Simulations with many transmitters, receivers, and/or large antenna arrays become otherwise quickly prohibitively complex.
```bash
l_min, l_max = time_lag_discrete_time_channel(rg.bandwidth)
l_tot = l_max-l_min+1

a, tau = cdl(batch_size=2, num_time_steps=rg.num_time_samples+l_tot-1, sampling_frequency=rg.bandwidth)
```
`rg.num_time_samples=1148=14x(76+6)`. For example, a is `[2, 1, 16, 1, 2, 23, 1164]`, where `1164` is from `14x(76+6)+17-1=1164`, where `rg.cyclic_prefix_length=6`.

```bash
h_time = cir_to_time_channel(rg.bandwidth, a, tau, l_min=l_min, l_max=l_max, normalize=True)
# Function that will apply the discrete-time channel impulse response to an input signal
channel_time = ApplyTimeChannel(rg.num_time_samples, l_tot=l_tot, add_awgn=True)
```
where `h_time` is Discrete-time channel impulse response `[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, l_max - l_min + 1]`, e.g., the shape is `[2, 1, 16, 1, 2, 1164, 17]`

During time-domain OFDM transmission: 
```bash
b = binary_source([batch_size, 1, rg.num_streams_per_tx, encoder.k])
c = encoder(b)
x = mapper(c)
x_rg = rg_mapper(x)
x_rg shape: (4, 1, 2, 14, 76)

# OFDM modulation with cyclic prefix insertion
x_time = modulator(x_rg)
x_time shape: (4, 1, 2, 1148)

# Compute the discrete-time channel impulse reponse
cir = cdl(batch_size, rg.num_time_samples+l_tot-1, rg.bandwidth)
h_time = cir_to_time_channel(rg.bandwidth, *cir, l_min, l_max, normalize=True)
h_time shape: `[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, l_max - l_min + 1]` (4, 1, 16, 1, 2, 1164, 17)

#channel output
y_time = channel_time([x_time, h_time, no])
y_time shape: (4, 1, 16, 1164)

# OFDM demodulation and cyclic prefix removal
y = demodulator(y_time)
y shape: (4, 1, 16, 14, 76)

#perfect_csi
a shape: `[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]` (4, 1, 16, 1, 2, 23, 1164)
tau shape: `[batch size, num_rx, num_tx, num_paths]` (4, 1, 1, 23)
# We need to sub-sample the channel impulse reponse to compute perfect CSI
# for the receiver as it only needs one channel realization per OFDM symbol
a_freq = a[...,rg.cyclic_prefix_length:-1:(rg.fft_size+rg.cyclic_prefix_length)] #6:-1:76+6, where rg.fft_size=76
a_freq = a_freq[...,:rg.num_ofdm_symbols]

a_freq shape: (4, 1, 16, 1, 2, 23, 14)
h_freq shape: (4, 1, 16, 1, 2, 14, 76)
h_hat shape: (4, 1, 16, 1, 2, 14, 64)
x_hat shape: (4, 1, 2, 768)
no_eff shape: (4, 1, 2, 768)
llr shape: (4, 1, 2, 1536)
b_hat shape: (4, 1, 2, 768)
BER: 0.0
```

### discrete-time impulse response (time-domain) in our code `deepMIMO5.py`
???? In the member function `def generateChannel(self, x_rg, no, channeltype='ofdm'):` of `class Transmitter` in `deepMIMO5.py`, it contains the `cir_to_time_channel` function and generate the discrete-time channel impulse reponse
```bash
h_time = cir_to_time_channel(bandwidth, h_b, tau_b, l_min=l_min, l_max=l_max, normalize=True) 
#h_time: [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, l_max - l_min + 1] complex[64, 1, 1, 1, 16, 1, 27]
channel_time = MyApplyTimeChannel(self.RESOURCE_GRID.num_time_samples, l_tot=l_tot, add_awgn=False)
y_time = channel_time([x_time, h_time]) #(64, 1, 1, 1090) complex
```
The plot of the discrete-time impulse reponse is shown:

![Discretetime CIR](../imgs/ofdm_discretetimeCIR.png)

## OFDM Processing:  `StreamManagement` and `ResourceGrid`

These DeepMIMO related code is in `class Transmitter():`, and the initialization is in `init` function side. In addition to these DeepMIMO related code, `init` function also contains the MIMO related code `StreamManagement` and `MyResourceGrid`
```bash
#NUM_STREAMS_PER_TX = NUM_UT_ANT
#NUM_UT_ANT = num_rx
num_streams_per_tx = num_rx ##1
RX_TX_ASSOCIATION = np.ones([num_rx, num_tx], int) #[[1]]
self.STREAM_MANAGEMENT = StreamManagement(RX_TX_ASSOCIATION, num_streams_per_tx) #RX_TX_ASSOCIATION, NUM_STREAMS_PER_TX
```
If no `Guard` is added, the resource grid is shown here with `RESOURCE_GRID.num_data_symbols=14(OFDM symbol)*76(subcarrier) array=1064` as the grid size and all 1064 grids are data.
![Resource Grid](../imgs/deepmimo_resourcegrid.png)


Codeword length is `1064*(num_bits_per_symbol = 4)=4256`, Number of information bits per codeword is also `k=4256` if LDPC is not used.
```bash
b = binary_source([self.batch_size, 1, self.num_streams_per_tx, self.k]) #if empty [64,1,1,4256] [batch_size, num_tx, num_streams_per_tx, num_databits]
x = self.mapper(b=c) #if empty np.array[64,1,1,1064] 1064*4=4256 [batch_size, num_tx, num_streams_per_tx, num_data_symbols]
```
After `mapper`, the bits information (4256bits) has been converted to `[64,1,1,1064]` means 1064 symbols (`num_bits_per_symbol = 4`). `rg_mapper` will map the 1064 symbols into the resource grid `14*76=1064`
```bash
x_rg = self.rg_mapper(x) ##array[64,1,1,14,76] 14*76=1064
#output: [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size][64,1,1,14,76]
```

If adding `Guard`, the resource grid shown here with `RESOURCE_GRID.num_data_symbols=14(OFDM symbol)*76(subcarrier) array=1064` as the grid size.
![Resource Grid](../imgs/ofdmresourcegrid.png)

The pilot pattern is shown here and 1064 grids contains the data, DC and pilot. `RESOURCE_GRID.num_data_symbols=768` instead of 1064. 
![PilotDataPattern](../imgs/ofdmdatapilot.png)


If chose the time-domain channel, i.e., `channeltype=="time"`, ResourceGrid bandwidth is `bandwidth= self.fft_size(76)*self.subcarrier_spacing=4560000`. `l_min, l_max = time_lag_discrete_time_channel(bandwidth) #-6, 20` computes the smallest and largest time-lag for the descrete complex baseband channel. The smallest time-lag returned is always -6. This value was deemed small enough for all models. The largest time-lag is computed from the `bandwidth` and `maximum_delay_spread` as follows: $ L_{\text{max}} = \lceil W \tau_{\text{max}} \rceil + 6 $, where: $L_{\text{max}}$ represents the largest time-lag, $W$ corresponds to the bandwidth, $\tau_{\text{max}}$ is the maximum delay spread. The default value for `maximum_delay_spread` is 3 microseconds (3us). This value was found to be large enough to include most significant paths with all channel models, assuming a nominal delay spread of 100 nanoseconds.

`cir_to_time_channel`: Compute the channel taps forming the discrete complex-baseband representation of the channel from the Channel Impulse Response (CIR) (``a``, ``tau``). The channel impulse response represents how a channel responds to an impulse (delta function) transmitted through it.
It characterizes the channel’s behavior over time, including multipath effects, delays, and attenuation.

The function of `cir_to_time_channel` assumes that a sinc filter is used for pulse shaping and receive filtering. Therefore, given a channel impulse response $(a_{m}(t), \tau_{m}), 0 \leq m \leq M-1$, the channel taps are computed as follows:
```math
\bar{h}_{b, \ell}
= \sum_{m=0}^{M-1} a_{m}\left(\frac{b}{W}\right)
   \text{sinc}\left( \ell - W\tau_{m} \right)
```
for $`\ell`$ ranging from $l_{min}$ to $l_{max}$, and where $W$ is the $bandwidth$. Each tap ($\bar{h}_{b, \ell}$) represents the combined effect of all paths at a specific time lag ($\ell$). The sinc function accounts for the time delay and phase shift due to each path. Input $a$ is Path coefficients: `[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], complex`, e.g., `(64, 1, 1, 1, 16, 10, 1)`. $tau$ is Path delays [s]: `[batch size, num_rx, num_tx, num_paths], float`, e.g., `(64, 1, 1, 10)`. Output $hm$ is Channel taps coefficients: `[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, l_max - l_min + 1], complex`, e.g., `[64, 1, 1, 1, 16, 1, 27]`.
The generated `h` is shown in this figure:
![Discrete Time CIR](../imgs/ofdm_discretetimeCIR.png)

## OFDM Transmission Code
### CDL and DeepMIMO Ttesting
`deeplearning/AIsim_main2.py` is created to perform complete OFDM transmission over CDL or DeepMIMO channel dataset. The major class is `class Transmitter()`.

`def test_CDLchannel()` inside the `deeplearning/AIsim_main2.py` is used to test the OFDM transmission over the CDL OFDM channel (`channeldataset='cdl'`), `channeltype='time'` and `channeltype='ofdm'` with `perfect_csi=False` and `perfect_csi=True` are tested inside this function. The dictionary data obtained after OFDM transmission is saved in `data/cdl_time_saved_ebno5.npy`, `data/cdl_time_saved_ebno5perfectcsi.npy`, `data/cdl_ofdm_saved_ebno5.npy`, and `data/cdl_ofdm_saved_ebno5perfectcsi.npy`.

`test_DeepMIMOchannel()` inside the `deeplearning/AIsim_main2.py` is used to test the OFDM transmission over the DeepMIMO OFDM channel (`channeldataset='deepmimo'`)


### BER Evaluation
`def sim_bersingle2` performs BER test for `channeldataset='deepmimo'` and `channeldataset='cdl'`, `channeltype='ofdm'` or `channeltype='time'`, and draws a single BER and BLER curve for one scenario. The following cases are tested with BER curves.
```bash
    bers, blers, BERs = sim_bersingle2(channeldataset='cdl', channeltype='time', NUM_BITS_PER_SYMBOL = 2, EBN0_DB_MIN = -5.0, EBN0_DB_MAX = 25.0, \
                   BATCH_SIZE = 32, NUM_UT = 1, NUM_BS = 1, NUM_UT_ANT = 2, NUM_BS_ANT = 16, showfigure = False, datapathbase='data/')
    bers, blers, BERs = sim_bersingle2(channeldataset='deepmimo', channeltype='time', NUM_BITS_PER_SYMBOL = 2, EBN0_DB_MIN = -5.0, EBN0_DB_MAX = 25.0, \
                   BATCH_SIZE = 32, NUM_UT = 1, NUM_BS = 1, NUM_UT_ANT = 1, NUM_BS_ANT = 16, showfigure = False, datapathbase='data/')
    bers, blers, BERs = sim_bersingle2(channeldataset='cdl', channeltype='ofdm', NUM_BITS_PER_SYMBOL = 2, EBN0_DB_MIN = -5.0, EBN0_DB_MAX = 25.0, \
                   BATCH_SIZE = 128, NUM_UT = 1, NUM_BS = 1, NUM_UT_ANT = 2, NUM_BS_ANT = 16, showfigure = False, datapathbase='data/')
    bers, blers, BERs = sim_bersingle2(channeldataset='deepmimo', channeltype='ofdm', NUM_BITS_PER_SYMBOL = 2, EBN0_DB_MIN = -5.0, EBN0_DB_MAX = 25.0, \
                   BATCH_SIZE = 128, NUM_UT = 1, NUM_BS = 1, NUM_UT_ANT = 1, NUM_BS_ANT = 16, showfigure = False, datapathbase='data/')
```

`sim_bermulti()` contains multiple BER testing cases and put many figures into one BER comparison graph.

### deepMIMO5 and AIsim Main2 comparison (for debug)
`deepMIMO5.py`:
ebno_db=5
h: (1, 1, 1, 16, 10, 1) tau: (1,1,10)
num_streams_per_tx=1
b: (64, 1, 1, 2288), k=2288
x_rg: (64, 1, 1, 14, 76) [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
print(y.shape) #[64, 1, 1, 14, 76] dim (3,4 removed) h_out: (64, 1, 1, 1, 16, 1, 44)
h_hat: (64, 1, 1, 1, 1, 14, 44) [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,fft_size]
x_hat: (64, 1, 1, 572)  [batch_size, num_tx, num_streams, num_data_symbols]
llr_est: (64, 1, 1, 2288) [batch size, num_rx, num_rx_ant, n * num_bits_per_symbol]
b_hat: (64, 1, 1, 2288)
BER Value: 0.2825

`AIsim_main2.py`:
self.num_time_steps = 1 #num_ofdm_symbols
ebno_db=5
h_b: (2, 1, 1, 1, 16, 10, 14), tau_b: (2, 1, 1, 10)
h_out: (2, 1, 1, 1, 16, 14, 76)
b: (2, 1, 1, 3072) k=3072, 768*4=3072 RESOURCE_GRID.num_data_symbols * num_bits_per_symbol
x_rg: (2, 1, 1, 14, 76) [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
y shape: (2, 1, 1, 14, 76) [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
x_hat: (2, 1, 1, 768) 
llr_est: (2, 1, 1, 3072)
b_hat: (2, 1, 1, 3072)
BER Value: 0.2317
Perfect_csi: BER Value: 0.08251953125

ApplyOFDMChannel error: (64, 1, 1, 1, 16, 1, 76)
#inputs x :  [batch size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], complex
#h_freq : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers], complex Channel frequency responses
#h_freq: (64, 1, 1, 1, 16, 1, 76)
ValueError: operands could not be broadcast together with shapes (64,1,1,1,16,1,76) (64,1,1,1,2,14,76) 
operands could not be broadcast together with shapes (64,1,1,1,16,1,76) (64,1,1,1,2,14,76) 

h:(64, 1, 1, 1, 16, 1, 76), x: (64, 1, 1, 1, 2, 14, 76)
(64, 1, 1, 1, 16, 1, 76), x: (64, 1, 1, 1, 2, 14, 76)

h_b: (64, 1, 1, 1, 16, 10, 1), tau_b: (64, 1, 1, 10) (64, 1, 1, 1, 16, 1, 76)



