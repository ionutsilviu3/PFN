# Brain MRI Tumor Detection

This project implements a Convolutional Neural Network (CNN) for brain tumor detection from MRI images, using imperative and functional styles in PyTorch and JAX, for an technical analysis between the three of them. The dataset consists of MRI scans classified as "Tumor" or "No Tumor".

## Structure

- `imperative.ipynb`, `functional.ipynb`, `jax_implementation.ipynb`: Main notebooks for training and evaluation.
- `run_jax.py`, `run_notebook.py`: Scripts for running experiments.
- `data/brain_tumor_dataset/`: Data

## Quick Start

1. **Create a virtual environment (Python 3.10)**  
    ```
    python -m venv venv
    venv\Scripts\activate
    ```

2. **Install dependencies**  
    ```
    pip install -r requirements.txt
    ```

3. **Install Docker**  
    Ensure [Docker](https://www.docker.com/get-started/) is installed and running on your system.

4. **Execute PyTorch notebooks N times**  
    ```
    python run_notebook.py imperative.ipynb 10
    python run_notebook.py functional.ipynb 10
    ```

5. **Run JAX version in Docker**  
    ```
    docker build -f Dockerfile.jax -t jax-brain-tumor:latest .
    docker run --rm --gpus all -v ${PWD}:/workspace -w /workspace jax-brain-tumor:latest python3.10 run_jax.py
    ```