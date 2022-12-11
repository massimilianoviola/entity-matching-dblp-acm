import argparse
import html
import matplotlib.pyplot as plt
import os
import pandas as pd
import string
from sentence_transformers import SentenceTransformer, util
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


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


def compute_metrics(true, pred):
    precision = precision_score(true, pred)
    recall = recall_score(true, pred)
    f1 = f1_score(true, pred)
    
    return precision, recall, f1


def title_match_accuracy(title_A, title_B):
    title_A = set(title_A.split())
    title_B = set(title_B.split())
    
    intersection = len(title_A & title_B)
    normalization_factor = min(len(title_A), len(title_B))
    
    return intersection / normalization_factor


def create_X_and_y(A, B, verbose=False):
    """Given two input dataframes A and B, generate the feature matrix and corresponding labels
    for the training phase by:
    - applying a year blocking scheme,
    - quickly removing candidates unlikely to match with title intersection,
    - retrieve labels using ground truth,
    - computing similarity measures between the two 'title' columns and the two 'authors'
    columns using embeddings from a language model.
    
    Returns
    -------
    X : feature matrix of (n_samples, 2)
        title and authors similarity measures from LM embeddings
    y : target vector (n_samples,) of 0/1
    """
    cartesian = A.merge(B, how="cross", suffixes=("_a", "_b"))
    # year blocking scheme
    cartesian = cartesian.loc[cartesian.year_a == cartesian.year_b]
    
    # remove candidates unlikely to match
    cartesian["title_match"] = cartesian.apply(
        lambda x: title_match_accuracy(x["title_a"], x["title_b"]), axis=1)
    cartesian = cartesian.loc[cartesian.title_match >= 0.5]
    
    # get corresponding targets by looking at the ground truth
    merge = pd.merge(cartesian, df_true, how="left", 
                     left_on=["id_a", "id_b"], right_on=["idACM", "idDBLP"]) \
              .loc[:, ["id_a", "title_a", "authors_a", "id_b", "title_b", "authors_b", "true"]] \
              .fillna(value={"true": 0}) \
              .sample(frac=1, ignore_index=True) # shuffle
    
    # calculate the cosine similarity of titles using embeddings from a LM
    if verbose:
        print("Extracting title embeddings...")
    emb_title_a = model.encode(merge.title_a, batch_size=16, show_progress_bar=verbose)
    emb_title_b = model.encode(merge.title_b, batch_size=16, show_progress_bar=verbose)
    merge["title_sim"] = util.dot_score(emb_title_a, emb_title_b).diag()
    
    # do the same for the author columns
    if verbose:
        print("Extracting authors embeddings...")
    emb_authors_a = model.encode(merge.authors_a, batch_size=16, show_progress_bar=verbose)
    emb_authors_b = model.encode(merge.authors_b, batch_size=16, show_progress_bar=verbose)
    merge["authors_sim"] = util.dot_score(emb_authors_a, emb_authors_b).diag()
    
    return merge.loc[:, ["title_sim", "authors_sim"]], merge.true


def create_X_and_ids(A, B, verbose=False):
    """Given two input dataframes A and B, generate candidate matches and their
    feature matrix at prediction time by:
    - applying a year blocking scheme,
    - quickly removing candidates unlikely to match with title intersection,
    - computing similarity measures between the two 'title' columns and the two 'authors'
    columns using embeddings from a language model.
    
    Returns
    -------
    X : feature matrix of (n_samples, 2)
        title and authors similarity measures from LM embeddings
    ids : dataframe of (n_samples, 2) ida, idb pairs
    """
    
    cartesian = A.merge(B, how="cross", suffixes=("_a", "_b"))
    # year blocking scheme
    cartesian = cartesian.loc[cartesian.year_a == cartesian.year_b]
    
    # stage 1: quickly remove candidates unlikely to match
    cartesian["title_match"] = cartesian.apply(
        lambda x: title_match_accuracy(x["title_a"], x["title_b"]), axis=1)
    candidates = cartesian.loc[cartesian.title_match >= 0.5]
    candidates = candidates.rename(columns={"id_b": "idDBLP", "id_a": "idACM"}) \
                           .reset_index(drop=True)
    
    # stage 2: use a more accurate language model as a feature extractor for pairs likely to match
    # calculate the cosine similarity of titles using embeddings from a LM
    if verbose:
        print("Extracting title embeddings...")
    emb_title_a = model.encode(candidates.title_a, batch_size=16, show_progress_bar=verbose)
    emb_title_b = model.encode(candidates.title_b, batch_size=16, show_progress_bar=verbose)
    candidates["title_sim"] = util.dot_score(emb_title_a, emb_title_b).diag()
    
    # do the same for the author columns
    if verbose:
        print("Extracting authors embeddings...")
    emb_authors_a = model.encode(candidates.authors_a, batch_size=16, show_progress_bar=verbose)
    emb_authors_b = model.encode(candidates.authors_b, batch_size=16, show_progress_bar=verbose)
    candidates["authors_sim"] = util.dot_score(emb_authors_a, emb_authors_b).diag()

    return candidates.loc[:, ["title_sim", "authors_sim"]], candidates.loc[:, ["idDBLP", "idACM"]]


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="NN", choices=["NN", "RF", "SVM"],
                        help="Model to predict test set: neural network (NN), random forest (RF), support vector machine (SVM)")
    parser.add_argument("--cross_validation", action="store_true",
                        help="Run cross-validation on the training dataframes")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Increase output verbosity, display progress bars and results")
    parser.add_argument("--save_plot", action="store_true",
                        help="Save the plot of the decision boundary of the trained model predicting the test set")
    args = parser.parse_args()
    

    dfa, dfb, df_true = load_and_preprocess_data()

    # leave a 25% test split out in both dataframes as test
    dfa_train, dfa_test = train_test_split(dfa, test_size=0.25, random_state=0)
    dfb_train, dfb_test = train_test_split(dfb, test_size=0.25, random_state=0)
    dfa_train = dfa_train.reset_index(drop=True)
    dfb_train = dfb_train.reset_index(drop=True)

    # load language model to be used as feature extractor
    model = SentenceTransformer("all-MiniLM-L12-v2")


    if args.cross_validation:
        if args.verbose:
            print("Running cross-validation...")
        
        # run 3-fold cross-validation on the train dataframes
        kf = KFold(n_splits=3, shuffle=True, random_state=0)

        for fold, ((train_idxa, valid_idxa), (train_idxb, valid_idxb)) in \
            enumerate(zip(kf.split(dfa_train), kf.split(dfb_train))):
            
            if args.verbose:
                print("#"*25)
                print(f"### Fold {fold + 1}")
            
            ta = dfa_train.loc[train_idxa]
            tb = dfb_train.loc[train_idxb]
            
            va = dfa_train.loc[valid_idxa]
            vb = dfb_train.loc[valid_idxb]
            
            # create train feature matrix and target
            X_train, y_train = create_X_and_y(ta, tb, verbose=args.verbose)
            
            # create valid feature matrix and ids
            X_valid, df_pred = create_X_and_ids(va, vb, verbose=args.verbose)
            
            if args.model == "RF":
                clf = RandomForestClassifier(n_estimators=200, max_depth=2, random_state=0)
            elif args.model == "SVM":
                clf = SVC(kernel="rbf")
            else:
                clf = MLPClassifier(hidden_layer_sizes=(8,), max_iter=400, random_state=0)

            clf.fit(X_train, y_train)

            pred = clf.predict(X_valid)
            df_pred["pred"] = pred
            
            # pairs in ground truth with elements in va and vb
            true_matches = df_true.loc[(df_true.idACM.isin(va.id)) & (df_true.idDBLP.isin(vb.id))]
            df_result = pd.merge(df_pred, true_matches, how="outer").fillna(0)
            precision, recall, f1 = compute_metrics(df_result.true, df_result.pred)
            
            # display fold statistics
            if args.verbose:
                print(f"### Matches in train: {len(df_true.loc[(df_true.idACM.isin(ta.id)) & (df_true.idDBLP.isin(tb.id))])}")
                print(f"### Matches in valid: {len(true_matches)}")
                print(f"### --Valid metrics-- precision: {precision:.3f}, recall: {recall:.3f}, f1_score: {f1:.3f}")
        
        if args.verbose:
            print("#"*25)
    
    
    if args.verbose:
        print("Training on the whole training set...")
    
    # train model with the whole training set
    X, y = create_X_and_y(dfa_train, dfb_train, verbose=args.verbose)

    if args.model == "RF":
        clf = RandomForestClassifier(n_estimators=200, max_depth=2, random_state=0)
    elif args.model == "SVM":
        clf = SVC(kernel="rbf")
    else:
        clf = MLPClassifier(hidden_layer_sizes=(8,), max_iter=400, random_state=0)

    clf.fit(X, y)
    
    if args.verbose:
        print("Predicting test set...")
    
    # predict test
    X_test, df_pred = create_X_and_ids(dfa_test, dfb_test, verbose=args.verbose)
    pred = clf.predict(X_test)
    df_pred["pred"] = pred
    
    # pairs in ground truth with elements in dfa_test and dfb_test
    true_matches = df_true.loc[(df_true.idACM.isin(dfa_test.id)) & (df_true.idDBLP.isin(dfb_test.id))]
    df_result = pd.merge(df_pred, true_matches, how="outer").fillna(0)
    precision, recall, f1 = compute_metrics(df_result.true, df_result.pred)
    
    if args.verbose:
        print(f"### Matches in test to predict: {len(true_matches)}")
        print(f"### --Test metrics-- precision: {precision:.3f}, recall: {recall:.3f}, f1_score: {f1:.3f}")
        
    
    if args.save_plot:
        # plot decision boundary of the classifier and train/test points
        disp = DecisionBoundaryDisplay.from_estimator(
            clf, X, response_method="predict",
            xlabel="Title cosine similarity", ylabel="Authors cosine similarity",
            alpha=0.6, cmap="RdYlGn")

        disp.ax_.scatter(X.loc[y==0, "title_sim"], X.loc[y==0, "authors_sim"],
                         c="r", edgecolor="k", label="Train non-match")
        disp.ax_.scatter(X.loc[y==1, "title_sim"], X.loc[y==1, "authors_sim"],
                         c="g", edgecolor="k", label="Train match")
        disp.ax_.scatter(X_test.loc[:, "title_sim"], X_test.loc[:, "authors_sim"],
                         c="orange", marker="x", label="Test")

        if args.model == "RF":
            name = "Random forest"
        elif args.model == "SVM":
            name = "SVM"
        else:
            name = "Neural network"        
        plt.title(f"{name} decision boundary")
        plt.xlim(-0.15, 1.05)
        plt.ylim(-0.15, 1.05)
        plt.legend()
        
        if not os.path.exists("plots"):
            os.mkdir("plots")
        plt.savefig(f"plots/{'_'.join(name.split())}_decision_boundary.png", dpi=200)
