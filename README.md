# Entity Matching
This file is a README for the semester exercise in Data Integration and Large Scale Analysis. The point of the exercise was to implement non machine learning and machine learning approaches to the entity matching of records from two databases. The implementation took place at TU Graz during the winter semestr 2022. 

## Authors
David Mihola, david.mihola@student.tugraz.at, 12211951  
Massimiliano Viola, massimiliano.viola@student.tugraz.at, 12213195

## Used data for entity matching
The used data can be downloaded using this link https://dbs.uni-leipzig.de/file/DBLP-ACM.zip. After downloading it, extract the content to the root directory of the project.

## Required libraries
Install the required libraries with `pip install -r ./requirements.txt` for both scripts.

## Scripts
We separated the *Task 01* and *Task 02* of the excercise into two scripts. The usage of the respective scripts is the following:
```
python ./task_01/entity-matching.py [-h] [-t {levenshtein,accuracy,exact}] [-s]

options:
  -h, --help            show this help message and exit
  -t {levenshtein,accuracy,exact}, --technique {levenshtein,accuracy,exact}
                        Title matching techniques: Levenshtein ration (levenshtein), Our accuracy measure (accuracy), Exact match (exact).
  -s, --save_plot       Save the plot of the decision boundary with scattered matches and non-macthes.

```
```
python ./task_02/ml-entity-matching.py [-h] [-m {NN,RF,SVM}] [-p {1,2,3}] [-c] [-v] [-s]

options:
  -h, --help            show this help message and exit
  -m {NN,RF,SVM}, --model {NN,RF,SVM}
                        Model to predict test set: neural network (NN), random forest (RF), support vector machine (SVM).
  -p {1,2,3}, --hyper_parameters {1,2,3}
                        Hyper parameters of the chosen model: 1st set of parameters (1), 2nd set of parameters (2), 3rd set of parameters (3).
  -c, --cross_validation
                        Run cross-validation on the training dataframes.
  -v, --verbose         Increase output verbosity, display progress bars and results.
  -s, --save_plot       Save the plot of the decision boundary of the trained model predicting the test set.
```

## Implementation details shared between non ML and ML approaches
In this section we describe, how we cleaned, pre-processed and blocked the records. For specific implementation details for each approach see the README files in directories `task_01/` and `task_02/` respectively.

### Cleaning and pre-processing
The only cleaning of the data was necessary for the `authors` column in the database `ACM.csv`, where we replaced HTML escaped characters by their actual UTF-8 values.

In the final code, we pre-processed only the columns `title` and `authors`, as we used only these two columns for the entity matching. 

#### Titles pre-processing
We pre-preocessed the records in the `title` column by applying these steps:
1. normalizing the values to lowercase,
2. replacing all punctuation characters with spaces,
3. replacing multiple successive spaces with a single space.

#### Authors pre-processing
We pre-processed the records in the `authors` column by applying these steps:
1. sorting the authors' names alphabetically.
 
### Blocking scheme
We used blocking of the records according to the `year` column. Although it is a very simple blocking scheme, it had a perfect resulting accuracy.

## Results
See the README files in directories `task_01/` and `task_02/` respectively for non ML approach and ML approach.

## Result reproduction
See the README files in directories `task_01/` and `task_02/` respectively for non ML approach and ML approach.
