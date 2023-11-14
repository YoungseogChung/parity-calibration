# Parity Calibration

Repository for [Parity Calibration](https://proceedings.mlr.press/v216/chung23a.html) (Published at UAI 2023, Oral)

This repository includes all of the functionalities presented in the paper and reproduction of the weather set of experiments.

## Installation
To use the code, we recommend using Python 3.8+ and creating a new environment, e.g.
```python
conda create --name parity_calibration python=3.8
conda activate parity_calibration
```

The requirements can be installed with pip:
```python
pip install -r requirements.txt
```

Lastly, the package can be installed, also with pip:
```python
pip install -e .
```

## Example Usage
You can refer to the file `experiments/run_weather_experiments.py` for an example of how to use the code.

The file can be run with the command
```python
python experiments/run_weather_experiments.py
```
which will run the weather prediction experiment with the default parameters set in this file.

## Citation
This repository was based on the following paper:
```
@InProceedings{pmlr-v216-chung23a,
  title = 	 {Parity calibration},
  author =       {Chung, Youngseog and Rumack, Aaron and Gupta, Chirag},
  booktitle = 	 {Proceedings of the Thirty-Ninth Conference on Uncertainty in Artificial Intelligence},
  pages = 	 {413--423},
  year = 	 {2023},
  editor = 	 {Evans, Robin J. and Shpitser, Ilya},
  volume = 	 {216},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {31 Jul--04 Aug},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v216/chung23a/chung23a.pdf},
  url = 	 {https://proceedings.mlr.press/v216/chung23a.html},
  abstract = 	 {In a sequential regression setting, a decision-maker may be primarily concerned with whether the future observation will increase or decrease compared to the current one, rather than the actual value of the future observation. In this context, we introduce the notion of parity calibration, which captures the goal of calibrated forecasting for the increase-decrease (or “parity") event in a timeseries. Parity probabilities can be extracted from a forecasted distribution for the output, but we show that such a strategy leads to theoretical unpredictability and poor practical performance. We then observe that although the original task was regression, parity calibration can be expressed as binary calibration. Drawing on this connection, we use an online binary calibration method to achieve parity calibration. We demonstrate the effectiveness of our approach on real-world case studies in epidemiology, weather forecasting, and model-based control in nuclear fusion.}
}
```

## Questions?
Please do not hesitate to reach out if you have any questions: [Youngseog Chung](https://github.com/YoungseogChung) (youngsec (at) cs.cmu.edu)


