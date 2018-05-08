import numpy as np
import os

path = "/Users/tao/Downloads/ok/temp/jena_climate_2009_2016.csv"

fh = open(path)
lines = fh.read().split("\n")
fh.close()

header = lines[0].split(",")

float_data = np.zeros((len(lines)-1, len(header)-1))
for i, line in enumerate(lines[1:]):
    values = [float(x) for x in line.split(",")[1:]]
    float_data[i, :] = values
N = 200000
mean = float_data[:N].mean(axis=0)
float_data -= mean
std = float_data[:N].std(axis=0)
float_data /= std
