# Task 01: Entity Matching Pipeline
This file is a README for the python script `ml-entity-matching.py`, where we implement machine learning approach to the entity matching of records from two databases.

## Implementation details
In this section we describe, how we matched the records.

### Hyper-parameters
We chose three different ML models, which are a neural network, a random forest and a support vector machine, and 3 different parameters for each of the models as the hyper parameters of our ML pipeline. 

### Records matching
To evaluate, if two records match, we used only the columns `title` and `authors`, as adding the `venue` column to our matching pipeline decreased its accuracy. Our approach to the matching of records is the following:
* TODO

## Results
We evaluated the results of our pipeline with three common accuracy metrics in information retrieval, i.e. **precision**, **recall** and **f1 score**.

### Best performing model based on the cross validation results
We evaluated the model performace as an average f1 score over the validations folds. The model with the best cross-validation performance is the support vector machine model with radial basis function kernel initialization. The results of this model on the test data set are:
* precision: **98.66 %**
* recall:    **98.66 %**
* f1 score:  **98.66 %**

Futhermore, the results of all the models on both the validation data sets and the test data set can be seen in the file `results.txt`.

#### Matches, non-matches and test samples scattered on the plot of the models decision boundary
![SVM1](../plots/SVM_1_decision_boundary.png)

## Result reproduction
Run the script from the root directory of the project after installing the requirements, downloading and extracting the data as described in the main README file. Use the command to reproduce the results:
* `python ./task_02/ml-entity-matching.py`   - to obtain the results of the best performing model regarding the cross validation on the test set
