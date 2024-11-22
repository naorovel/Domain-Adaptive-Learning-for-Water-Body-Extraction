# Domain-Adaptive Learning for Water Body Extraction

## Setup
All dependencies for this repository are dockerized. This makes it easy to run the repository without dealing with dependency issues. 

To create the environment needed to run this repository, please install and start Docker first. 

If running the repository in VSCode, install the following extension pack: 
- Remote Development (ms-vscode-remote.vscode-remote-extensionpack)

To build and run the Docker container: 
- `cmd + shift + P` `"Build and Run Container"`

If VSCode asks for config file options, use the default settings. 

In the terminal of the container, activate the conda environment.  
`conda activate dalwbe`

The conda environment currently contains the following core dependencies: 
```
- gdal/osr/osgeo
- numpy
- rasterio
- pytorch-lightning
- pillow
```

It is easiest to work on the repository inside of the Docker container to avoid dependency clashes. 