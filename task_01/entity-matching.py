import argparse
import html
import Levenshtein
import matplotlib.pyplot as plt
import os
import pandas as pd
import string
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import precision_score, recall_score, f1_score


def preprocess_titles(df):
    """Normalize titles by converting to lowercase,
    replacing punctuation characters with spaces,
    and removing double spaces.
    """
    return df.title.str.lower() \
                   .str.translate(str.maketrans(string.punctuation, " " * len(string.punctuation))) \
                   .map(lambda x: " ".join(x.split()))


def preprocess_authors(df):
    """Preprocess authors by filling in the missing values,
    converting html escaped latin characters,
    and sorting authors alphabetically.
    """
    return df.authors.fillna("") \
             .map(lambda x: html.unescape(x)) \
             .map(lambda x: ", ".join(sorted(x.split(", "))))


def load_and_preprocess_data():
    dfa = pd.read_csv("DBLP-ACM/ACM.csv")
    dfa.loc[:, "title"] = preprocess_titles(dfa)
    dfa.loc[:, "authors"] = preprocess_authors(dfa)

    dfb = pd.read_csv("DBLP-ACM/DBLP2.csv", encoding="latin_1")
    dfb.loc[:, "title"] = preprocess_titles(dfb)
    dfb.loc[:, "authors"] = preprocess_authors(dfb)
    
    df_true = pd.read_csv("DBLP-ACM/DBLP-ACM_perfectMapping.csv")
    df_true["true"] = 1

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


def create_X_and_y(dfa, dfb, df_true, title_camparrison_func, author_camparrison_func):
    year_blocked = dfa.merge(dfb, right_on="year", left_on="year") # blocking the data in the files by year
    year_blocked = (
        pd.merge(year_blocked, df_true, how="left", left_on=["id_x", "id_y"], right_on=["idACM", "idDBLP"]) # join with ground truth
          .loc[:, ["title_x", "authors_x", "title_y", "authors_y", "true"]] # select only necessary columns
          .fillna(value={"true": 0}) # mark non-matches
          .sample(frac=1, ignore_index=True) # shuffle
    )

    year_blocked["title_sim"] = year_blocked.apply(
        lambda x: title_camparrison_func(x["title_x"], x["title_y"]), axis=1
    ) # calculate the desired title similarity for each title pair
    year_blocked["authors_sim"] = year_blocked.apply(
        lambda x: author_camparrison_func(x["authors_x"], x["authors_y"]), axis=1
    ) # calculate the desired title similarity for each authors pair

    return year_blocked.loc[:, ["title_sim", "authors_sim"]], year_blocked.true


def calculate_stats(predicted, ground_truth):
    precision = precision_score(ground_truth, predicted)
    recall = recall_score(ground_truth, predicted)
    f1 = f1_score(ground_truth, predicted)
    return precision, recall, f1


class Classifier():
    def __init__(self):
        self.fit = None # to be usable in DecisionBoundaryDisplay

    def prediction_condition(self, title_sim, author_sim):
        if title_sim >= 0.95: # titles match almost completely
            return True

        elif (title_sim >= 0.85 and # titles match still well, but another comfimation is needed
              author_sim >= 0.5):   # record A has at least half of the authors of record B or vice versa
            return True                              
        
        elif (title_sim >= 0.75 and # title match is not so strong, stronger confirmation is needed
              author_sim >= 1.0):   # record A has all authors of record B or vice versa
            return True
        
        return False # Not a match

    def predict(self, X):
        return self.predict_df(X).to_numpy()
    
    def predict_df(self, X):
        return X.apply(lambda x: self.prediction_condition(x["title_sim"], x["authors_sim"]), axis=1)
    
    def __sklearn_is_fitted__(*_): # to be usable in DecisionBoundaryDisplay
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--technique", default="levenshtein", choices=["levenshtein", "accuracy", "exact"],
                        help="Title matching techniques: Levenshtein ratio (levenshtein), Our accuracy measure (accuracy), Exact match (exact).")
    parser.add_argument("-s", "--save_plot", action="store_true",
                        help="Save the plot of the decision boundary with scattered matches and non-matches.")
    args = parser.parse_args()

    name = "Levenshtein ratio"
    title_camparrison_func = lambda x, y: Levenshtein.ratio(x, y, score_cutoff=0.75) # normalized similarity thresholded at 0.75
    if args.technique == "exact":
        title_camparrison_func = lambda x, y: x == y
        name = "Exact match"
    elif args.technique == "accuracy":
        title_camparrison_func = title_match_accuracy
        name = "Our accuracy measure"

    dfa, dfb, df_true = load_and_preprocess_data()

    X, y = create_X_and_y(dfa, dfb, df_true, title_camparrison_func, authors_match_accuracy)
    clf = Classifier()

    predicted = clf.predict_df(X)
    precision, recall, f1 = calculate_stats(predicted, y)

    print("Accuracy metrics:")
    print(f"precision: {precision:.4f}", f"recall:    {recall:.4f}", f"f1 score:  {f1:.4f}", sep="\n")

    if args.save_plot:
        # plot decision boundary of the classifier and classified points
        disp = DecisionBoundaryDisplay.from_estimator(clf, X, response_method="predict", xlabel="Title similarity", 
                                                      ylabel="Authors similarity", alpha=0.6, cmap="RdYlGn")
        disp.ax_.scatter(X.loc[y==0, "title_sim"], X.loc[y==0, "authors_sim"], c="r", edgecolor="k", label="Non-match")
        disp.ax_.scatter(X.loc[y==1, "title_sim"], X.loc[y==1, "authors_sim"], c="g", edgecolor="k", label="Match")
        plt.xlim(-0.15, 1.05)
        plt.ylim(-0.15, 1.05)
        plt.title(f"{name} decision boundary")
        plt.legend(loc="upper left")

        if not os.path.exists("plots"):
            os.mkdir("plots")
        plt.savefig(f"plots/{args.technique}_decision_boundary.png", dpi=200)
