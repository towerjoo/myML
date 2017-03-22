
class Perceptron:
    def __init__(self, input_number, activator):
        self.activator = activator
        self.weights = [0.] * input_number
        self.bias = 0.

    def __str__(self):
        return "weights\t: {}\nbias\t: {}".format(self.weights, self.bias)

    def predict(self, input_vec):
        return self.activator(
                reduce(lambda x, y: x+y,
                    map(lambda (w, x): w * x, 
                        zip(self.weights, input_vec)),
                        0.)  + self.bias
                )

    def train(self, input_vecs, labels, iteration, rate):
        sample = zip(input_vecs, labels)
        for _ in range(iteration):
            for input_vec, label in sample:
                output = self.predict(input_vec)
                self.update_weights(input_vec, output, label, rate)

    def update_weights(self, input_vec, y, label, rate):
        self.weights = map(lambda (w, x): w + rate * (label - y)*x,
                            zip(self.weights, input_vec))
        self.bias = self.bias + rate * (label - y)

def f(x):
    return 1 if x > 0 else 0

def get_training_set_and():
    input_vecs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    labels = [0, 0, 0, 1]
    return input_vecs, labels

def get_training_set_or():
    input_vecs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    labels = [0, 1, 1, 1]
    return input_vecs, labels


def train_perceptron(get_training_set):
    input_vecs, labels = get_training_set()
    input_num = len(input_vecs[0])
    N = 10
    rate = 0.1 # learning rate
    p = Perceptron(input_num, f)
    p.train(input_vecs, labels, N, rate)
    return p

if __name__ == "__main__":
    perceptrons = {
        "AND": get_training_set_and,
        "OR": get_training_set_or,
    }
    for k, training_set_func in perceptrons.iteritems():
        print "{} perceptron".format(k)
        p = train_perceptron(training_set_func)
        print p
        print '1 {} 1 = {}'.format(k, p.predict([1, 1]))
        print '0 {} 0 = {}'.format(k, p.predict([0, 0]))
        print '1 {} 0 = {}'.format(k, p.predict([1, 0]))
        print '0 {} 1 = {}'.format(k, p.predict([0, 1]))
