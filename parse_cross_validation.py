import re
import numpy as np

file = open("results_cv.txt", "r")
data = file.read()

names = re.findall(r"Test accuracy metrics for (.*?):", data)

data = data.split("fold 1:")
f1_sums = []
for model_run in data[1:]:
    values = re.findall(r"f1 score:  (.{6})", model_run)
    f1_sums.append(float(values[0]) + float(values[1]) + float(values[2]))

print(f1_sums)
print(names[np.array(f1_sums).argmax()])
