# Task 01: Entity Matching Pipeline
This file is a README for the python script `ml-entity-matching.py`, where we implement machine learning approach to the entity matching of records from two databases.

## Implementation details
In this section we describe, how we matched the records.

### Records matching
To evaluate, if two records match, we used only the columns `title` and `authors`, as adding the `venue` column to our matching pipeline decreased its accuracy. Our approach to the matching of records is the following:
TODO

## Results
We evaluated the results of our pipeline with three common accuracy metrics in information retrieval, i.e. **precision**, **recall** and **f1 score**, which nicely summerize the accuray of the pipeline.

### Best performing ML model based on the validation set results
* precision: **98.66 %**
* recall:    **98.66 %**
* f1 score:  **98.66 %**

## Result reproduction
Install the required libraries with `pip install -r ./task_02/requirements.txt`. Run the script from the root directory of the project after downloading and extracting the data as described in the main README file.

The commands to run the script and reproduce the results listed above are:
* `python ./task_02/ml-entity-matching.py --save_plot -m NN`    - for matching using a neural network
* `python ./task_02/ml-entity-matching.py --save_plot -m RF`    - for matching using a random forest
* `python ./task_02/ml-entity-matching.py --save_plot -m SVM`   - for matching using a support vector machine
