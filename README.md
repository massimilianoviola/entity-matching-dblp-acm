# Entity Matching
This repository contains a solution to the exercise for the *Data Integration and Large Scale Analysis* course that took place at TU Graz during the winter semester of 2022. The point of the exercise is to implement first a non-machine learning (Task 01) and then a machine learning (Task 02) approach to the problem of entity matching of records from two databases.

## Authors
David Mihola, david.mihola@student.tugraz.at, 12211951 and Massimiliano Viola, massimiliano.viola@student.tugraz.at, 12213195

## Used data for entity matching
The data used for this project is the DBLP-ACM dataset, containing bibliographic records with `title`, `authors`, `venue`, and `year` attributes to match across two different tables. The data can be downloaded from the scource https://dbs.uni-leipzig.de/file/DBLP-ACM.zip. After downloading the zip file, extract the content to the root directory of the project.

## Required libraries
All the necessary libraries to run both tasks can be installed with `pip install -r requirements.txt`. A working installation of Python is required as a prerequisite.

## Scripts
We separate the *Task 01* and *Task 02* of the exercise into two scripts. The usage of the respective scripts from the root directory is the following:
```
python ./task_01/entity-matching.py [-h] [-t {levenshtein,accuracy,exact}] [-s]

optional arguments:
  -h, --help            show this help message and exit
  -t {levenshtein,accuracy,exact}, --technique {levenshtein,accuracy,exact}
                        Title matching techniques: Levenshtein ratio (levenshtein), Our accuracy measure (accuracy), Exact match (exact).
  -s, --save_plot       Save the plot of the decision boundary with scattered matches and non-matches.

```
```
python ./task_02/ml-entity-matching.py [-h] [-m {NN,RF,SVM}] [-p {1,2,3}] [-c] [-v] [-s]

optional arguments:
  -h, --help            show this help message and exit
  -m {NN,RF,SVM}, --model {NN,RF,SVM}
                        Model to predict test set: neural network (NN), random forest (RF), support vector machine (SVM).
  -p {1,2,3}, --hyper_parameters {1,2,3}
                        Hyper-parameters of the chosen model: 1st set of parameters (1), 2nd set of parameters (2), 3rd set of parameters (3).
  -c, --cross_validation
                        Run cross-validation on the training dataframes.
  -v, --verbose         Increase output verbosity, display progress bars and results.
  -s, --save_plot       Save the plot of the decision boundary of the trained model predicting the test set.
```

## Implementation details shared between non-ML and ML approaches
In this section, we describe how we apply a blocking scheme, clean, and pre-process the records. For specific implementation details for each approach, please see the README files in directories `task_01/` and `task_02/` respectively.

### Blocking scheme
We use blocking of the records according to the `year` column. Although it is a very simple blocking scheme, it has a perfect recall of the matches and reduces the number of comparisons at prediction time by an order of magnitude.

### Cleaning and pre-processing
The only necessary cleaning operation of the data is for the `authors` column in the database `ACM.csv`. HTML escaped characters are replaced by their actual UTF-8 values to avoid mismatches due to different encodings only.

In the final code, the columns `title` and `authors` are the only ones used to make predictions in the entity matching pipeline. These two columns are pre-processing as follows:

#### Titles pre-processing
We pre-process the `title` column by applying these steps:
1. normalizing the values by converting them to lowercase,
2. replacing all punctuation characters with spaces,
3. replacing multiple successive spaces with a single space.

#### Authors pre-processing
We pre-process the `authors` column by applying these steps:
1. filling in the missing values,
2. sorting the authors' names alphabetically.

## Results and results reproduction
See the README files in directories `task_01/` and `task_02/` respectively for the non-ML approach and ML approach.
