# Surfalex data explorer

This repository contains interactive analysis to supplement our manuscript : '***A novel integrated framework for reproducible formability predictions using virtual materials testing***', *A. J. Plowman, P. Jedrasiak, T. Jailin, P. Crowther, S. Mishra, P. Shanthraj, J. Quinta da Fonseca*, submitted to Materials Open Research.

The manuscript describes the development of a new computational framework for formability studies, [MatFlow](https://github.com/LightForm-group/matflow). We apply this framework to study the formability of the Surfalex HF (AA6016A) alloy.

In this repository, we include Jupyter notebooks that can be used to interactively explore the experimental data. Data and analysis from the DIC, EBSD, and forming limit experiments can be examined. We also include a Jupyter notebook to explore the five MatFlow workflows that we developed in support of this work.

## Getting started

To run the Jupyter notebooks, clone this repository, then activate a new Python 3 virtual environment (tested on Python 3.8) from the root of the cloned repository with:

`python -m venv venv`

Activate this environment and then install the required packages using:

`python -m pip install -r requirements.txt`

After this, you can start Jupyter with the command: `jupyter notebook`.
