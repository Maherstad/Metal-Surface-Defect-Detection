# Metal Surface Defect Detection - Examples

This directory contains Jupyter notebooks demonstrating various aspects of the project.

## Notebooks

### EDA.ipynb
Exploratory Data Analysis of the metal surface defect dataset:
- Dataset statistics and distribution
- Class balance analysis
- Defect type visualizations
- Bounding box analysis

### real_time_detection.ipynb
Real-time inference demonstrations:
- Loading trained models
- Running inference on images and videos
- Visualization of detection results
- Performance benchmarking

### package_simulation.ipynb
Package development and simulation:
- Module structure testing
- API design validation
- Integration examples

## Usage

Start Jupyter Notebook:

```bash
jupyter notebook examples/
```

Or use VS Code's Jupyter extension to open and run the notebooks directly.

## Prerequisites

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
pip install jupyter
```

## Data Requirements

Some notebooks require the dataset to be downloaded. Run the setup script first:

```bash
python setup.py
```
