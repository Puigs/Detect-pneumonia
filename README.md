# T-DEV-810 : zoidberg2.0

This project was carried out by : 
Brayan Puigsagur: For the CNN part
Clément L.: For the pre-processing part
Gabriel B.: For the KNN part
Manon V.: For the SVM part

## Introduction

Given some X-ray images, we will use machine learning to help doctors detecting pneumonia.

## Getting started

### Requirements

- [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

### Managing environments

Conda allows you to create separate environments containing files, packages, and their dependencies that will not interact with other environments.

1. Create a new environment and install a package in it.

We will name the environment zoidberg and install all the libraries and package of the IA.yml file. At the Anaconda Prompt or in your terminal window, type the following:
```bash
conda env create --file IA.yml -n zoidberg
```

2. To use, or "activate" the new environment, type the following:

```bash
conda activate zoidberg
```

3. To see a list of all your environments, type:

```bash
conda info --envs
```

4. Change your current environment back to the default (base): 

```bash
conda activate
```

## The Models :

- [Supervised Learning](SVM/)
