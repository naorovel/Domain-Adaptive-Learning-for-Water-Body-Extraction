# Domain-Adaptive Learning for Water Body Extraction

## Setup
This repository is Dockerized - meaning that all dependencies are contained inside a Docker container. This makes it easy to run and install necessary dependencies for this repository without worrying about version management between team members. 

If us

To build the Docker image, run the following in the terminal: 
`docker build -t domain_adaptive_water_body_extraction .`

To run the code through the main entrypoint in the Docker container: 
`docker run -p 5000:5000 domain_adaptive_water_body_extraction`

