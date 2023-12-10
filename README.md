# ML Engineering Assignment

## Description

This project contains a collection of notebooks for data analysis and machine learning.

- AttentionCaching - Pix2Struct.ipynb: This notebook contains the code for the Pix2Struct model.
- DocVQA Reader Task.ipynb: This notebook contains the code for the DocVQA Reader Task.

## Requirements

To run the notebooks in this project, you need to have the following requirements installed:

- Python (version 3.8.18)
- Additional Python packages (listed in requirements.txt)
- Java: OpenJDK Runtime Environment Homebrew (build 11.0.21+0)
- SBT (version 1.9.7)

## Setup

1. Clone the repository
2. Create a virtual environment
3. ```pip install -r requirements.txt```
4. Download the [DocVQA dataset](https://rrc.cvc.uab.es/?ch=17&com=downloads) and extract it to the ```data``` folder. It should follow the following structure:
```
data
└───DocVQA
    └───documents
    └───ocr
    └───test_v1.0.json
    └───train_v1.0_withQT.json
    └───val_v1.0_withQT.json
```
where documents contains the images.
5. ```cd vqareader; sbt clean package; cp target/scala-2.12/docvqareader_2.12-0.1.jar ../```

## Usage

```jupyter notebook```

and open the notebooks in the browser.

## Where to find the code

All the implementation required for the task of AttentionCaching - Pix2Struct is placed in ```inference.py``` file.

The implementation for DocVQAReader Task is placed in multiple places:

- ```vqareader``` contains the Scala implementations of the DocVQAReader.
- ```jsl``` contains the Python wrapper for the Scala implementation.
