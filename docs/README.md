# Texera

## Confusion Matrix Chart

This operator is used to plot the Confusion Matrix.

The Confusion Matrix is applied to evaluate the quality of the output of a classifier on the dataset. The diagonal elements represent the number of points for which the predicted label is equal to the true label, while off-diagonal elements are those that are mislabeled by the classifier. The higher the diagonal values of the confusion matrix the better, indicating many correct predictions.

More details about [Confusion Matrix](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#confusion-matrix).

### Parameters

| Name            | Description                                 |
| --------------- | ------------------------------------------- |
| Plot Title      | The title of the chart.                     |
| Actual Value    | The column of ground truth.                 |
| Predicted Value | The attribute of the predicted value column |

### Output

A visualization of Confusion Matrix.

## ROC Chart

ROC curves typically feature a true positive rate (TPR) on the Y axis and a false positive rate (FPR) on the X axis. This means that the top left corner of the plot is the “ideal” point - a FPR of zero, and a TPR of one. This is not very realistic, but it does mean that a larger area under the curve (AUC) is usually better. The “steepness” of ROC curves is also important since it is ideal to maximize the TPR while minimizing the FPR.

ROC curves are typically used in binary classification, where the TPR and FPR can be defined unambiguously. In the case of multiclass classification, a notion of TPR or FPR is obtained only after binarizing the output. This can be done in 2 different ways:

- the One-vs-Rest scheme compares each class against all the others (assumed as one);
- the One-vs-One scheme compares every unique pairwise combination of classes.

More details about [ROC](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py).

### Parameters

| Name                          | Description                                     |
| ----------------------------- | ----------------------------------------------- |
| Plot Title                    | The title of the chart.                         |
| Actual Value                  | The column of ground truth.                     |
| Predicted Probabilities Value | The attribute of the probabilities value column |

### Output

A visualization of ROC.

## Category to Number

In machine learning, converting strings into enumerated types (numerical values) through a process often referred to as encoding is crucial for several reasons:

1. **Algorithm Compatibility**: Most machine learning algorithms are designed to operate on numerical data, meaning they require inputs to be in numerical form. String or textual data often contain categorical information (e.g., gender, country) that is incompatible for direct use in algorithms because these algorithms cannot perform mathematical operations on non-numeric data.
2. **Data Scaling**: Converting strings to numbers allows for data to be scaled (or normalized), which is crucial for many algorithms. For instance, when using distance-based algorithms (like K-Nearest Neighbors or Linear Regression), the scale of different features can greatly affect model performance. Through encoding and scaling, all features are transformed into numerical forms that can be compared against each other.
3. **Feature Representation**: Some encoding techniques, such as One-Hot Encoding, help algorithms better understand and differentiate between categories by converting each category into a unique binary vector. This representation eliminates any assumed ordinal relationship between categories, providing a clear way to distinguish between them.
4. **Efficiency**: Transforming textual data into enumerated types also increases processing efficiency. Numerical data typically require less storage space than string data and are more computationally efficient to process.
5. **Model Performance**: Proper data preprocessing and encoding strategies can significantly improve model performance. By converting strings to numbers, models can more accurately capture patterns in the data, often leading to better predictive outcomes.

For example: we assign a unique integer to each category. The encoding might look like this:

- Pet Type: Dog = 0, Cat = 1
- Pet Color: Black = 0, White = 1, Brown = 2

After encoding, our dataset becomes:

| Pet Type | Pet Color |
| -------- | --------- |
| 0        | 0         |
| 1        | 1         |
| 0        | 2         |
| 1        | 0         |

So, this operator is used to do Encoding.

### Parameters

| Name   | Description                              |
| ------ | ---------------------------------------- |
| Column | Select some columns you want to encoding |

### Output

If you choose a column named A, this operator will create a column called A_to_number, and output the encoding result in this new column. Other columns will not make any changes.
