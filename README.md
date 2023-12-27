# Stroke Prediction Model

## Introduction
This repository hosts a machine learning project designed to predict the likelihood of a stroke based on various health indicators and demographic information. The project encapsulates the entire workflow of a typical data science project including data exploration, preprocessing, model training, evaluation, and suggestions for improvement.

## Data Exploration
We initiated the project by performing an extensive exploratory data analysis (EDA). The process involved:
- Examining summary statistics and distributions of various features.
- Identifying missing values, particularly in the 'bmi' feature.
- Visualizing data distributions and potential correlations between features.

## Data Preprocessing
Data preprocessing was a critical step due to issues such as missing values and the need to convert categorical variables into a machine-readable format. The following steps were taken:
- Missing values in 'bmi' were imputed using the median of the feature.
- Categorical variables were encoded using label encoding.
- Numerical features were scaled to have a mean of zero and a standard deviation of one.

## Modeling
We employed a step-wise approach to model building:
1. **Logistic Regression**: Served as our baseline model.
2. **Random Forest**: Provided a more robust model capable of capturing non-linear patterns.
3. **Gradient Boosting and XGBoost**: Leveraged for their prowess in dealing with imbalanced datasets.

## Evaluation Metrics Explained
Model performance was evaluated using several metrics suited for imbalanced datasets:
- **Precision**: The accuracy of positive predictions.
- **Recall**: The ability of the model to capture actual positive instances.
- **F1-Score**: A balance between precision and recall.
- **Support**: The number of instances for each class in the validation set.
- **Accuracy**: Although not the primary metric due to class imbalance, it was still considered.
- **Macro Average**: Average performance across classes.
- **Weighted Average**: Average performance weighted by the number of instances in each class.

## Handling Imbalanced Data
To address the imbalance in the dataset, we explored:
- **SMOTE**: For oversampling the minority class.
- **Class Weight Adjustment**: To make the model pay more attention to the minority class.

## Results and Discussion
The models' predictions and their respective performance metrics were analyzed to identify the best-performing model. The analysis revealed a strong bias towards the majority class, prompting the use of SMOTE and class weight adjustments.

## Future Work
Future improvements could include:
- Advanced feature engineering.
- Hyperparameter optimization.
- Exploration of alternative resampling strategies.

## Installation and Usage
To replicate this project, clone the repository and install the required dependencies listed in `requirements.txt`.

```bash
git clone https://github.com/mayankbaluni/StrokeRiskPredictor.git
cd stroke-prediction
pip install -r requirements.txt

# End of script
exit 0
```

## Contact
For any queries or suggestions, feel free to contact me at [mayankbaluni@gmail.com]
