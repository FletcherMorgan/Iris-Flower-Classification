# Iris Flower Classification


## Overview

This project explores the classic Iris flower dataset to predict the species of an iris flower—*setosa*, *versicolor*, or *virginica*—using four easy-to-measure features:

1. Sepal length (cm)
2. Sepal width (cm)
3. Petal length (cm)
4. Petal width (cm)

Since the true species labels are known, this is a supervised, three-class classification problem. The main goals are:

- **Separability:** Determine if these features can effectively distinguish the three species.
- **Model Choice:** Evaluate logistic regression against simple, rule-based approaches.
- **Generalisation:** Assess model performance on unseen data.

Our end-goal is to build a model that is both **accurate** and **consistent** in classifying iris species.


## Project Structure

- **Exploratory Data Analysis (EDA):** Statistical summaries and visualizations (histograms, scatterplots, pairplots) to understand feature distributions and relationships with the target.
- **Data Preparation:** Loading the dataset from scikit-learn, converting to a DataFrame, and mapping targets to species names.
- **Train/Test Split:** Splitting the data into training and test sets to ensure robust model evaluation.
- **Modeling:** Building and evaluating classification models.


## Environment & Library Setup

- **Jupyter Notebooks** for interactive analysis.
- **Python 3** with libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn

Install dependencies as needed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```


## Data Loading & Preparation

- The dataset is loaded from `sklearn.datasets`.
- Features and targets are organized into a pandas DataFrame.
- Target integers are mapped to their corresponding species names for interpretability.


## Exploratory Data Analysis

- **Descriptive Statistics:** Summary metrics (mean, std, min, max) for each feature.
- **Univariate Analysis:** Histograms show distributions and potential separability.
- **Bivariate Analysis:** Scatterplots and pairplots visualize relationships between features and class labels.

**Key findings:**
- Petal length and petal width are highly informative, showing clear separation between species.
- Sepal features show more overlap, especially between *versicolor* and *virginica*.
- The setosa species is linearly separable based on petal measurements.


## Modeling Workflow

1. **Train/Test Split:** The data is split (80% train, 20% test) using `train_test_split` for unbiased evaluation.
2. **Model Training:** Models such as logistic regression are trained and compared.
3. **Evaluation:** Accuracy and generalization are assessed on the test set.


## Results & Insights

- The EDA shows that petal measurements are highly effective at distinguishing species.
- Logistic regression and even simple rule-based models can achieve high accuracy given the feature separability.
- Visualizations confirm that *setosa* is easily separated, while *versicolor* and *virginica* have some overlap.


## References

- The Iris dataset was introduced by R.A. Fisher and is widely used in pattern recognition literature.
- For more details, see the [scikit-learn dataset description](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset) and the references therein.


## Running the Notebook

1. Open `Iris_Flower_Classification.ipynb` in Jupyter Notebook or JupyterLab.
2. Run cells sequentially to reproduce the analysis, visualizations, and modeling steps.


## License

See the repository for license details.
