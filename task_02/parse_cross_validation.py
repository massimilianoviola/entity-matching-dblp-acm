import re
import numpy as np


with open("results_cv.txt", "r") as file:
    data = file.read()

names = re.findall(r"Test accuracy metrics for (.*?):", data)

data = data.split("fold 1:")
f1_sums = []
for model_run in data[1:]:
    values = re.findall(r"f1 score:  (.{6})", model_run)
    f1_sums.append(float(values[0]) + float(values[1]) + float(values[2]))

f1_sums = np.array(f1_sums)
with np.printoptions(precision=4, suppress=True):
    print(f"avg f1 cv scores: {f1_sums/3}")
    print(f"best model: {names[f1_sums.argmax()]} with avg f1 cv score of {f1_sums.max()/3:.4f}")
