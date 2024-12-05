# Domain-Adaptive Learning for Water Body Extraction

## Setup
All dependencies for this repository are included in the `environment.yaml` file. This makes it easy to run the repository without dealing with dependency issues. 

To create the environment needed to run this repository, please install the necessary dependencies as a `conda` environment using the following: 

`conda env create --f environment.yaml`

In the terminal of the container, activate the conda environment.  

`conda activate DALWBE`

The conda environment currently contains the following core dependencies: 
```
- numpy
- rasterio
- pytorch-lightning
- pillow
```