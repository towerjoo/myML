import math
num_S = sum([26 ** i for i in range(1, 27)])
print num_S
def Ai(i):
    p = 1
    for j in range(1, i+1):
        p = p * (26 - i + 1) 
    return float(p)


total = 0
for i in range(1, 27):
    total += Ai(i)
    print total

print float(total) / num_S, 1 / math.e

