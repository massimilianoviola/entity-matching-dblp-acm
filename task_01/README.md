# Task 01: Entity Matching Pipeline
This file is a README for the python script `entity-matching.py`, where we implement non machine learning approach to the entity matching of records from two databases.

## Implementation details
In this section we describe, how we matched the records.

### Records matching
To evaluate, if two records match, we used only the columns `title` and `authors`, as adding the `venue` column to our matching pipeline decreased its accuracy. Our approach to the matching of records is the following:
1. We calculated how well the titles of two records match, for which we used one of the three different techniquest described below.
2. Based on the title match accuracy, we either declare these records as matching, if the accuracy is higher or equal than 95 %, or as a not matching, if the accuracy is lower then 75 %.
3. When the title match accuracy is between 75 and 95 %, we additionaly use the authors match accuracy, see *Our accuracy* is section *Title matching techniques*, to determine, if the records are matching or not matching. When the title match accuracy is between 85 and 95 %, we required at least 50% match of the authors and for 75 and 85 % title match accuracy, we required a perfect match with regards to the authors.

#### Title matching techniques
* **Exact match**: This technique gives 100% accuracy, if the two input strings are equal, otherwise it gives 0% accuracy.
* **Our accuracy measure**: Our technique is inspired by intersection over union. We split the two input strings into words and make sets out of the words. The accuracy is then calculated as a fraction of the magnitude of the intersection of the two sets and the magnitude of the smaller set.
* **Levenshtein ration**: This technique is based on the levenshtein distance. It is calculated as a fraction of the difference between the lenght of the longer string and the levenshtein distance between the two input strings and the lenght of the longer string, or simplyfied as $1 - levenshtein\_distance / longer\_string\_length$.

## Results
We evaluated the results of our pipeline with three common accuracy metrics in information retrieval, i.e. **precision**, **recall** and **f1 score**, which nicely summerize the accuray of the pipeline.

### Title matching using levenshtein ratio
TODO
* precision: **97.23 %**
* recall:    **97.71 %**
* f1 score:  **97.47 %**

### Title matching using our title match accuracy
TODO
* precision: **94.21 %**
* recall:    **98.79 %**
* f1 score:  **96.44 %**

### Title matching using exact title match
TODO
* precision: **97.97 %**
* recall:    **91.05 %**
* f1 score:  **94.38 %**

## Result reproduction
Install the required libraries with `pip install -r ./task_01/requirements.txt`. Run the script from the root directory of the project after downloading and extracting the data as described in the main README file.

The commands to run the script and reproduce the results listed above are:
* `python ./task_01/entity-matching.py -st levenshtein`  - for the title matching using levenshtein ratio 
* `python ./task_01/entity-matching.py -st accuracy`     - for the title matching using our title match accuracy
* `python ./task_01/entity-matching.py -st exact`        - for the title matching using exact title match
