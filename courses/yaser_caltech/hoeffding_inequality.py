import random

def one_run():
    N = 1000
    m = 10
    v1 = 0
    vrand = 0
    vmin = 0
    min_count = m 
    first_count = 0
    rand_index = random.randint(0, N-1)
    rand_count = 0
    for i in range(N):
        head_count = 0
        for j in range(m):
            head_count += 1 if random.random() >= .5 else 0 # >= .5 means head
        if i == 0:
            first_count = head_count
        if min_count > head_count:
            min_count = head_count
        if i == rand_index:
            rand_count = head_count
    v1 = first_count / float(m)
    vrand = rand_count / float(m)
    vmin = min_count / float(m)
    return v1, vrand, vmin

N = 10000
v1_sum = 0
vrand_sum = 0
vmin_sum = 0
for i in range(N):
    v1, vrand, vmin = one_run()
    v1_sum += v1
    vrand_sum += vrand
    vmin_sum += vmin
print v1_sum/float(N), vrand_sum/float(N), vmin_sum / float(N)


