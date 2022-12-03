import pandas as pd                                                     # pip install pandas
from sklearn.metrics import precision_score, recall_score, f1_score     # pip install scikit-learn
import string   
import html 
import Levenshtein                                                      # pip install python-Levenshtein
import sys

def preprocess_titles(df):
    return (
        df.title.str.lower() # normalize to lower case
                .str.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))) # replace punctuation characters with spaces
                .map(lambda x: ' '.join(x.split())) # remove double spaces
    )

def preprocess_authors(df):
    return (
         df.loc[df.authors.notna(), 'authors'] # select only rows with some authors
           .map(lambda x: html.unescape(x)) # convert html escaped latin characters
           .map(lambda x: ', '.join(sorted(x.split(', ')))) # sort authors alphabetically
    )

def load_and_preprocess_data():
    df_true = pd.read_csv('DBLP-ACM/DBLP-ACM_perfectMapping.csv')
    df_true['true'] = 1

    dfa = pd.read_csv('DBLP-ACM/ACM.csv')
    dfa = dfa.set_index('id', drop=True)
    dfa.loc[:, 'title'] = preprocess_titles(dfa)
    dfa.loc[dfa.authors.notna(), 'authors'] = preprocess_authors(dfa)

    dfb = pd.read_csv('DBLP-ACM/DBLP2.csv', encoding='latin_1')
    dfb = dfb.set_index('id', drop=True)
    dfb.loc[:, 'title'] = preprocess_titles(dfb)
    dfb.loc[dfb.authors.notna(), 'authors'] = preprocess_authors(dfb)

    return dfa, dfb, df_true

def title_match_accuracy(title_A, title_B):
    title_A = set(title_A.split(" "))
    title_B = set(title_B.split(" "))

    intersection = title_A & title_B
    normalization_factor = min(len(title_A), len(title_B))
    return len(intersection) / normalization_factor

def authors_match_accuracy(authors_A, authors_B):
    authors_A = set(str(authors_A).split(", "))
    authors_B = set(str(authors_B).split(", "))

    intersection = authors_A & authors_B
    normalization_factor = min(len(authors_A), len(authors_B))
    return len(intersection) / normalization_factor

def blocking_scheme(dfa, dfb):
    for year in dfa.year.unique(): # blocking the data by year
        yield dfa.loc[dfa.year == year, ["title", "authors"]], dfb.loc[dfb.year == year, ["title", "authors"]]

def df_from_matches(matches):
    df = pd.DataFrame(matches, columns=['idDBLP', 'idACM'])
    df['pred'] = 1
    return df

def calculate_stats(df):
    precision = precision_score(df.true, df.pred) * 100
    recall = recall_score(df.true, df.pred) * 100
    f1 = f1_score(df.true, df.pred) * 100
    return precision, recall, f1

if __name__ == "__main__":
    title_camparrison_func = lambda x, y: Levenshtein.ratio(x, y, score_cutoff=0.75)

    if len(sys.argv) > 1: # argument was passed on the command line
        if sys.argv[1].lower() == "exact":
            title_camparrison_func = lambda x, y: x == y
        elif sys.argv[1].lower() == "accuracy":
            title_camparrison_func = title_match_accuracy
        elif sys.argv[1].lower() != "levenshtein":
            print("Wrong script argument.", "Try 'exact', 'accuracy' or 'levenshtein'.", file=sys.stderr, sep='\n')
            exit(1) # error

    dfa, dfb, df_true = load_and_preprocess_data()

    matches = []
    for dfa_block, dfb_block in blocking_scheme(dfa, dfb):
        for index_a, (title_a, author_a) in dfa_block.iterrows(): # with all records from first block
            for index_b, (title_b, author_b) in dfb_block.iterrows(): # comapare to all records from second block
                title_match = title_camparrison_func(title_a, title_b) # selected comparrison by the script argument

                if title_match >= 0.95: # titles match almost completely
                    matches.append([index_b, index_a])
                elif (title_match >= 0.85 and # titles match still well, but another comfimation is needed
                      authors_match_accuracy(author_a, author_b) >= 0.5): # record A has at least half of the authors of record B or vice versa
                    matches.append([index_b, index_a])
                elif (title_match >= 0.75 and # title match is not so strong, stronger confirmation is needed
                      authors_match_accuracy(author_a, author_b)) >= 1.0: # record A has all authors of record B or vice versa
                    matches.append([index_b, index_a])

    df_pred = df_from_matches(matches)
    df_result = pd.merge(df_pred, df_true, how='outer').fillna(0) # outer merge/join and Na replacement with zeros adds false positives and false negatives 
                                                                  # to the data frame

    precision, recall, f1 = calculate_stats(df_result)
    print(f"precision: {precision:.2f} %", f"recall:    {recall:.2f} %", f"f1 score:  {f1:.2f} %", sep='\n')
