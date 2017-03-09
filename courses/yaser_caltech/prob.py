def prob(p, n):
    pout = 0
    for i in range(0, n):
        pout += p * pow(1-p, i)
    return pout

if __name__ == "__main__":
    p = pow(0.45, 10)
    n = 1000
    print p
    print prob(p, n)
