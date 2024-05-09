# Flight Delay Prediction

This repository contains a machine learning model and associated code for predicting flight delays. The model is trained on historical flight data and can be used to predict whether a given flight will be delayed or not, based on various features such as airline, origin and destination airports, scheduled departure time, and more. This work is an expansion of that done as part of the final project for the Statistics and Modeling course at the University of Waterloo (2023).

## Features

- **Predictive Model**: A machine learning model trained on a large dataset of historical flight data, capable of predicting flight delays with high accuracy.
- **Data Preprocessing**: Scripts for cleaning and preprocessing raw flight data, handling missing values, and encoding categorical features.
- **Feature Engineering**: Code for extracting relevant features from the raw data, such as calculating the distance between airports, encoding airport and airline information, and more.
- **Model Training**: Scripts for training the machine learning model on the preprocessed data, with options for tuning hyperparameters and selecting the best-performing model.
- **Model Evaluation**: Code for evaluating the trained model's performance on a held-out test set, including metrics such as accuracy, precision, recall, and F1-score.
- **Deployment**: Scripts and instructions for deploying the trained model as a web service or API, allowing real-time prediction of flight delays.

## Getting Started

1. Clone the repository: `git clone https://github.com/username/flight-delay-prediction.git`
2. Preprocess the data: `python eda.py`
3. Train the model: `python model.py`


## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or bug fixes.

## License

This project is licensed under the [MIT License](LICENSE).

## TODO
Organize the code better by distributing it into appropriate directories suited for CI/CD.