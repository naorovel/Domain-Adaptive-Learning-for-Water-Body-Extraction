# Use an official GDAL image as the base image
FROM ghcr.io/osgeo/gdal:ubuntu-small-latest

# install pip
RUN apt-get update && apt-get -y install python3-pip --fix-missing \
    && apt-get install -y wget \
    && apt install -y git

# Install anaconda
ENV CONDA_DIR /opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/root/miniconda3/bin:$PATH"
RUN mkdir /root/.conda && bash Miniconda3-latest-Linux-x86_64.sh -b

# create conda environment
RUN conda init bash \
    && . ~/.bashrc \
    && conda create --name dalwbe python=3.8 \
    && conda activate dalwbe \
    && conda install gdal \
    && conda upgrade numpy \
    && conda install conda-forge::rasterio \
    && conda install conda-forge::pytorch-lightning \
    && conda install anaconda::pillow

# Set the working directory in the container
WORKDIR /app
