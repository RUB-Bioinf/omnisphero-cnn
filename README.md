# Omnisphero-CNN

*Neurosphere cultures consisting of primary human neural stem/progenitor cells (hNPC) are used for studying the effects of substances on early neurodevelopmental processes in vitro.
Differentiating hNPCs migrate and differentiate into radial glia, neurons, astrocytes, and oligodendrocytes upon plating on a suitable extracellular matrix and thus model processes of early neural development.
In order to characterize alterations in hNPC development, it is thus an essential task to reliably identify the cell type of each migrated cell in the migration area of a neurosphere.
To this end, we introduce and validate a deep learning approach for identifying and quantifying cell types in microscopic images of differentiated hNPC.
As we demonstrate, our approach performs with high accuracy and is robust against typical potential confounders.
We demonstrate that our deep learning approach reproduces the dose responses of well-established developmental neurotoxic compounds and controls, indicating its potential in medium or high throughput in vitro screening studies.
Hence, our approach can be used for studying compound effects on neural differentiation processes in an automated and unbiased process.*

***

Learn more: https://doi.org/10.1002/cyto.a.24514

***

[![GitHub stars](https://img.shields.io/github/stars/RUB-Bioinf/omnisphero-cnn.svg?style=social&label=Star)](https://github.com/RUB-Bioinf/omnisphero-cnn) 
[![GitHub forks](https://img.shields.io/github/forks/RUB-Bioinf/omnisphero-cnn.svg?style=social&label=Fork)](https://github.com/RUB-Bioinf/omnisphero-cnn)

***

[![Generic badge](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](docs/contribute.md)
![Size](https://img.shields.io/github/repo-size/RUB-Bioinf/omnisphero-cnn?style=plastic)
[![Language](https://img.shields.io/github/languages/top/RUB-Bioinf/omnisphero-cnn?style=plastic)](https://github.com/RUB-Bioinf/omnisphero-cnn)

***

[![Follow us on Twitter](https://img.shields.io/twitter/follow/NilsFoer?style=social&logo=twitter)](https://twitter.com/intent/follow?screen_name=NilsFoer)

[![Follow us on Twitter](https://img.shields.io/twitter/follow/JoshuaButke?style=social&logo=twitter)](https://twitter.com/intent/follow?screen_name=JoshuaButke)

***

[![Total alerts](https://img.shields.io/lgtm/alerts/g/RUB-Bioinf/omnisphero-cnn.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/RUB-Bioinf/omnisphero-cnn/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/RUB-Bioinf/omnisphero-cnn.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/RUB-Bioinf/omnisphero-cnn/context:python)

## Overview

This repository contains a few scripts to train, validate and save CNN models for the prediction of Neurons or / and Oligos on Neurospheres.
Inputs are defined as RGB images from antibody stained tissue data captured via fluorescence microscopy.

This pipeline can predict many datasets in a high throughput fashion.

***

![Example screenshot #1](/img/approach_overview.png?raw=true "Approach Overview")

## Usage
Learn more [here](https://github.com/RUB-Bioinf/omnisphero-cnn/wiki/Usage).

#### Training
To train a new model, use `train_model.py`.

- Give a comma seperated path list for training data which needs to be completly annotated
- Give a comma seperated path list for validation data which ALSO needs to be completly annotated
- Define the CNN model as you wish
- Set saving paths for the trained model, its weights and plots

#### Predicting
To predict data using a pre-trained model, use `predict_batch.py`.

- Give the location of a saved model which should be used to predict on new data
- Give the directory which contains unannotated data that should be labeled

#### CLI Predicting
To make batch-prediction easy, you can use the built-in CLI.
This function requires a pre-trained model and a batch of data, being located in a common root directory.

The CLI tool will prompt you to input all relevant variables and methods.
Run the CLI interface using:

```$ python predict_batch_custom.py```

This is the recommended way of running predictions if you do not want to change the sourcecode.

## Example Data
Find information on example data [here](https://github.com/RUB-Bioinf/omnisphero-cnn/wiki/Example-Data).

## Required libraries
To run this code, certain libraries are required.
Please aquire the these on your own, as they are not included within this repository:

 - numpy
 - pandas
 - keras
 - matplotlib
 - h5py
 - sklearn
 - imblearn

#### Conda Environment
If you do not want to set up these libraries yourself, you can use *Anaconda*.
Please look at the `/envs/` directory within this repository for the *Anaconda Environment* requirements file to recreate the environment used within this codebase.

## Cite

You can cite this work using this *BibTeX* entry:

```
@article{forsterreliable,
  title={Reliable Identification and Quantification of Neural Cells in Microscopic Images of Neurospheres},
  author={F{\"o}rster, Nils and Butke, Joshua and Ke{\ss}el, Hagen Eike and Bendt, Farina and Pahl, Melanie and Li, Lu and Fan, Xiaohui and Leung, Ping-chung and Klose, J{\"o}rdis and Masjosthusmann, Stefan and others},
  journal={Cytometry Part A},
  publisher={Wiley Online Library}
}
```

## Contact
Contact us on our homepage via:
http://www.bioinf.rub.de/contact/index.html.en

