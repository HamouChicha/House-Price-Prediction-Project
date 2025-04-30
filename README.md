# House Price Prediction

## Overview
This project aims to predict house prices using a dataset from Kaggle. It includes various stages of data processing, model building, and evaluation using multiple machine learning regression algorithms.

## Table of Contents
- [Project Description](## Project Description)
- [Dataset](## Dataset)
- [Files Included](## Files Included)
- [Prerequisites](## Prerequisites)
- [Usage](## Usage)
- [Model Evaluation](## Modeln Evaluation)
- [Algorithms Used](## Algorithms Used)

## Project Description
The House Price Prediction project involves:
- Data preprocessing and cleaning
- Label Encoding
- Feature selection
- Feature Scaling
- Model building and evaluation

The goal is to develop a predictive model that can estimate house prices based on various features, utilizing different regression algorithms to determine the best-performing model.

## Dataset
The dataset used for this project is sourced from Kaggle and is included as `data.csv`. It contains various features related to house attributes and their corresponding prices.

## Files Included
- `House_Price_Prediction.ipynb`: The main Jupyter Notebook containing the analysis and model.
- `functions.py`: A Python file with reusable functions utilized in the main notebook.
- `data.csv`: The dataset used for training the model.
- `description.txt`: A text file providing a description of the dataset.
- `requirements.txt`: A text file specifying the required packages for the project.

## Prerequisites
- Python 3.11.9
- Required libraries specified in `requirements.txt`

## Usage
To run the project, open the `House_Price_Prediction.ipynb` file in any Jupyter environment using the appropriate kernel and execute the cells sequentially.

## Model Evaluation
The performance of the trained models is evaluated using metrics such as Mean Absolute Error (MAE) and R-squared. The evaluation helps in determining which regression algorithm performs best for this dataset.

## Algorithms Used
Several machine learning regression algorithms were trained and evaluated, including:
- Linear Regression
- Random Forest Regression
- Gradient Boosting Regressor