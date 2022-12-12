# Entity Matching
This file is a README for the semester exercise in Data Integration and Large Scale Analysis. The point of the exercise was to implement non machine learning and machine learning approaches to the entity matching of records from two databases. The implementation took place at TU Graz during the winter semestr 2022. 

## Authors
David Mihola, david.mihola@student.tugraz.at, 12211951  
Massimiliano Viola, massimiliano.viola@student.tugraz.at, 12213195

## Used data for entity matching
The used data can be downloaded using this link https://dbs.uni-leipzig.de/file/DBLP-ACM.zip. After downloading it, extract the content to the root directory of the project.

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
