import re
import numpy as np

file = open("results_cv.txt", "r")
data = file.read()

names = re.findall(r"Test accuracy metrics for (.*?):", data)

data = data.split("fold 1:")
f1_averages = []
for model_run in data[1:]:
    values = re.findall(r"f1 score:  (.{6})", model_run)
    f1_averages.append(float(values[0]) + float(values[1]) + float(values[2]))

print(names[np.array(f1_averages).argmax()])
