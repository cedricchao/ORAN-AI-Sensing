# # Load a miniconda setup for our base Docker image which contains Python
# FROM continuumio/miniconda3

# # Install all necessary libraries
# RUN apt-get update && apt-get -y install build-essential musl-dev libjpeg-dev zlib1g-dev libgl1-mesa-dev wget dpkg libsctp-dev

# # Copy all the files in the current directory to /tmp/ml in our Docker image
# COPY . /tmp/ml

# # Go to /tmp/ml
# WORKDIR /tmp/ml

# # Install requirements.txt
# RUN pip install --upgrade pip && pip install requirements.txt

# # Set our xApp to run immediately when deployed
# ENV PYTHONUNBUFFERED 1
# CMD app.py

# Base image with Miniconda
FROM continuumio/miniconda3

# Install necessary system-level libraries
RUN apt-get update && apt-get -y install \
    build-essential \
    musl-dev \
    libjpeg-dev \
    zlib1g-dev \
    libgl1-mesa-dev \
    wget \
    libsctp-dev \
    lksctp-tools && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /tmp/ml

# Copy all files from the current directory to the container
COPY . /tmp/ml

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install git+https://github.com/philpraxis/pysctp.git && \
    pip install -r requirements.txt

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED 1

# Set the default command to run the application
CMD ["python3", "app.py"]
