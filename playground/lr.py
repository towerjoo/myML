import random

N = 178

test_N = int(178 * 0.2)
train_N = N - test_N

train_x = open("train_x.csv", "w")
train_y = open("train_y.csv", "w")
test_x = open("test_x.csv", "w")

trains_x = []
trains_y = []
tests = []

for line in open("wine.data"):
    if random.random() > 0.2:
        label = line.split(",")[0]
        data = ",".join(line.split(",")[1:])
        trains_x.append(data)
        trains_y.append(label)
    else:
        tests.append(",".join(line.split(",")[1:]))
train_x.write("".join(trains_x))
train_y.write("\n".join(trains_y))
test_x.write("".join(tests))
