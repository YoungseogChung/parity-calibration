import setuptools

setuptools.setup(
    name="parity_calibration",
    version="1.0.0",
    author="Youngseog Chung, Aaron Rumack, Chirag Gupta",
    author_email="youngsec@cs.cmu.edu",
    description=("Parity Calibration"),
    url="https://github.com/YoungseogChung/parity-calibration",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
