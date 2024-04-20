# Texera


## K-Nearest Neighbor Trainer

This operator is used to train the dataset in KNN model. There are two modes for this operator, which are KNN classifier and regressor, and they have same input ports, output ports, and parameter.  This operator aims to train the model, and pass the model information to the next operator.

### Input port

**Dataset**

Training dataset

**Parameter**

Parameter of model

### Output port

**Table**

Concat the dataset and model information in binary form

| Name                | Type                 | Description                      |
| ------------------- | -------------------- | -------------------------------- |
| Model               | Binary(Pickle.dumps) | Training model                   |
| Parameters          | Binary(Pickle.dumps) | Hyperparamters                   |
| Features            | Binary(Pickle.dumps) | Features used to train the model |
| Iteration(Optional) | Integer              |                                  |

### **Parameters**

There are two ways to pass the parameter to the model, user-define or optimization. 

- **Use optimization**
  
    Check the way passing the parameters
    
- **Column with class labels**
  
    Select column to be used as classification/regressor attribute.
    
- **Columns with features**
  
    Select column to be used as features
    
- **Number of neighbors to consider (k)**
  
    Select the number of nearest neighbors used to classify a new instance.
    

## Support Vector Machine

This operator is used to train the dataset in SVM model. There are two modes for this operator, which are SVM classifier and regressor, and they have same input ports, output ports, and parameter.  This operator aims to train the model, and pass the model information to the next operator.

### Input port

**Dataset**

Training dataset

**Parameter**

Parameter of model

### Output port

**Table**

Concat the dataset and model information in binary form

### **Parameters**

There are two ways to pass the parameter to the model, user-define or optimization. 

- **Use optimization**
  
    Check the way passing the parameters
    
- **Column with class labels**
  
    Select column to be used as classification/regressor attribute.
    
- **Columns with features**
  
    Select column to be used as features
    
- **Value of regularization parameter to consider (c)**
  
    Select the number of regularization parameter used to classify a new instance.
    
- **Kernel type**
  
    There are a number of kernels to choose from. Each kernel has its own parameters, which appear in the configuration dialog just under the kernel.
    
- **Value of degree to consider (optional)**
  
    Select the degree of the ‘poly’ kernel function.
    
- **Value of gamma to consider (optional)**
  
    Select the gamma of the ‘rbf’, ‘poly’ and ‘sigmoid’ kernel function.
    
- **Value of coef0 to consider (optional)**
  
    Select the coef0 of the ‘poly’ and ‘sigmoid’ kernel function.
    

| Kernel type | degree | gamma | coef |
| --- | --- | --- | --- |
| linear |  |  |  |
| poly | Y | Y | Y |
| rbf |  | Y |  |
| sigmoid |  | Y | Y |

## Apply Model

This operator is used to apply trained model to new datasets.  
A model is first trained on a training dataset by another Operator, which is often a learning algorithm implemented by sci-kit learn and whose name ends with trainer. Afterwards, this model can be applied on another datasets. Usually, the goal is to get a prediction on unseen data.  
The testing dataset upon which the model is applied, has to be compatible with the Attributes of the model. This means that the testing dataset has to include the attributes used to train this model in the previous trainer operator.         


### Input Port
**Model**  
Trained by upstream trainer
The model was trained by scikit-learn and saved with pickle.  
Including information of the features that used to train the model in column “features”  
Including the parameter of the model in column “para”.  

**Dataset**  
The testing dataset that will be performed prediction on with the given model.  
The testing dataset has to include the attributes used to train this model.  


### Output Port

All the attributes from model input port will be passed to output port.  
The predicted value will be in the output port, the name of this attribute can be assigned by the user, and the default value will be “y_pred”.  
If the user chooses “Ground Truth In Datasets”, the ground truth values will be passed to the output port.   
If the user chooses “Predict Probability For Each Class”, the model will predict the probability of one dataset belonging to each class. And the name of this attribute can be assigned by the user, and the default value will be “y_prob”. (not apply to regression model)  

| Name                                          | Type                 | Description                      |
| --------------------------------------------- | -------------------- | -------------------------------- |
| Model                                         | Binary(Pickle.dumps) | Training model                   |
| Parameters                                    | Binary(Pickle.dumps) | Hyperparamters                   |
| Features                                      | Binary(Pickle.dumps) | Features used to train the model |
| y_pred                                        | Binary(Pickle.dumps) | Predict values                   |
| y_porb(Optional)                              | Binary(Pickle.dumps) | Predicted probability column     |
| GroudTruthLabel(Name based on user selection) | Binary(Pickle.dumps) | Ground Truth                     |
| Iteration(Optional)                           | Integer              |                                  |

## Scorer
The Scorer operator is designed to assess model performance through various metrics, tailored to address both regression and classification challenges. It is structured to have uniform input ports, output ports, and parameters for both problem types.

### Input Port
The input to the Scorer operator is derived from the schema output by the Apply Model operator. Users must specify both the predicted value column and the actual value column. Additionally, users are required to select the metrics by which they wish to evaluate model performance.

### Output Port
The output of the Scorer operator retains all attributes from the input while modifying the format of the "para" attribute for enhanced clarity. Furthermore, the output table combines the results of all selected metrics, providing an overview of model performance. 

| Name                                          | Type                 | Description                      |
| --------------------------------------------- | -------------------- | -------------------------------- |
| Model                                         | Binary(Pickle.dumps) | Training model                   |
| Parameters                                    | String               | Hyperparamters                   |
| Features                                      | Binary(Pickle.dumps) | Features used to train the model |
| y_pred                                        | Binary(Pickle.dumps) | Predict values                   |
| y_porb(Optional)                              | Binary(Pickle.dumps) | Predicted probability column     |
| GroudTruthLabel(Name based on user selection) | Binary(Pickle.dumps) | Ground Truth                     |
| Iteration(Optional)                           | Integer              |                                  |
| Label                                         | String               | Each Predict value               |
| Accuracy(Optional)                            | Double               | Accuracy                         |
| Precision Score(Optional)                     | Double               | Accuracy                         |
| Recall Score(Optional)                        | Double               | Accuracy                         |
| F1 Score(Optional)                            | Double               | Accuracy                         |
|                                               |                      |                                  |
|                                               |                      |                                  |

## Model Selection
This operator is used to choose the best models from model tuning loops based on the chosen metric.    
It is always being used after loop end control block.  
The scorer operator has to be included in its upstream operators.

### Input Port

The schema from input port need to include the metrics calculated by scorer operator for either regression tasks or classification tasks.

### Output Port

All attributes from the input port will be transmitted to the output port.   
The user selects a metric from the input datasets, and optimal models are chosen based on which prediction yields superior performance. Evaluation of performance is conducted using the metric selected by the user from the input schema. The output will consist of the filtered result based on the minimum or maximum value among the chosen metric.

| Name                                          | Type                 | Description                      |
| --------------------------------------------- | -------------------- | -------------------------------- |
| Model                                         | Binary(Pickle.dumps) | Training model                   |
| Parameters                                    | String               | Hyperparamters                   |
| Features                                      | Binary(Pickle.dumps) | Features used to train the model |
| y_pred                                        | Binary(Pickle.dumps) | Predict values                   |
| y_porb(Optional)                              | Binary(Pickle.dumps) | Predicted probability column     |
| GroudTruthLabel(Name based on user selection) | Binary(Pickle.dumps) | Ground Truth                     |
| Iteration(Optional)                           | Integer              |                                  |
| Label                                         | String               | Each Predict value               |
| Accuracy(Optional)                            | Double               | Accuracy                         |
| Precision Score(Optional)                     | Double               | Accuracy                         |
| Recall Score(Optional)                        | Double               | Accuracy                         |
| F1 Score(Optional)                            | Double               | Accuracy                         |

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

### Input Port

Input one model with the predicted value attribute.

### Output Port

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

### Input Port

Input one model with the predicted probabilities attribute.

### Output Port

A visualization of ROC.

## Category to Number (Deprecated)

In machine learning, converting strings into enumerated types (numerical values) through a process often referred to as encoding is crucial for several reasons:

1. **Algorithm Compatibility**: Most machine learning algorithms are designed to operate on numerical data, meaning they require inputs to be in numerical form. String or textual data often contain categorical information (e.g., gender, country) that is incompatible with direct use in algorithms because these algorithms cannot perform mathematical operations on non-numeric data.
2. **Efficiency**: Transforming textual data into enumerated types also increases processing efficiency. Numerical data typically require less storage space than string data and are more computationally efficient to process.
3. **Model Performance**: Proper data preprocessing and encoding strategies can significantly improve model performance. By converting strings to numbers, models can more accurately capture patterns in the data, often leading to better predictive outcomes.

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

### Input Port

Input a dataset.

### Output Port

If you choose a column named A, this operator will create a column called A_to_number, and output the encoding result in this new column. Other columns will not make any changes.
