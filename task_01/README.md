# Task 01: Entity Matching Pipeline
This file is a README for the python script `entity_matching.py`, where we implement non machine learning approach to the entity matching of records from two databases. The implementation took place at TU Graz during the winter semestr 2022.

## Authors
David Mihola, david.mihola@student.tugraz.at, 12211951  
Massimiliano Viola, massimiliano.viola@student.tugraz.at, 12213195

## Implementation details
In this section we describe, how we cleaned, pre-processed, blocked and compared the records.

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
We used blocking of the records according to the `year` column. Although it is a very simple blocking scheme, it had the best (TODO even perfect ???) resulting accuracy.

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
We evaluated the results of our pipeline with three common metrics in information retrieval, i.e. **precision**, **recall** and **f1 score**, which nicely summerize the accuray of the pipeline.

### Levenshtein ratio title matching
The levenshtein ratio title matching has the overall best accuracy. It yields the best compromise between precision and recall.
* precision: **97.23 %**
* recall:    **97.71 %**
* f1 score:  **97.47 %**

### Our title match accuracy measure matching
Our title match accuracy measure matching has still very good overall accuracy. It slightly favors recall over accuracy.
* precision: **94.21 %**
* recall:    **98.79 %**
* f1 score:  **96.44 %**

### Exact title match matching
The exact title match matching has the worst overall accuracy of our approaches, but the accuracy is still very good, considering it is the simples possible matching scheme, which does not even consider the `authors` column. It favors accuracy over recall.
* precision: **97.97 %**
* recall:    **91.05 %**
* f1 score:  **94.38 %**

## Result reproduction
The script must be run from a directory, which contains a directory `DBLP-ACM` containing the database files `ACM.csv`, `DBLP-ACM_perfectMapping.csv` and `DBLP2.csv`. Regarding the used non standart python libraries, see the code, how to install them.

The commands to run the script and reproduce the results listed above are:
* `python entity_matching.py levenshtein`  - for the levenshtein ratio title matching
* `python entity_matching.py accuracy`     - for our title match accuracy measure matching
* `python entity_matching.py exact`        - for exact title match matching
