import numpy as np

class SoftTree:
    def __init__(self, X, Y, V, R, type='r'):
        assert type == 'r' or type == 'c'
        self.type = type
        self.root = self.Node()

        self.root.w = np.zeros(X.shape[1])
        self.root.w0 = 0
        self.root.w0 = np.mean(Y)

        self.root.splitNode(X, Y, V, R, self)

    def evaluate(self, x):
        if self.type == 'r':
            return self.root.evaluate(x)
        elif self.type == 'c':
            return self.sigmoid(self.root.evaluate(x))

    def mean_sq_err(self, X, Y):
        assert self.type == 'r'
        err = 0
        for i in range(len(Y)):
            y = self.evaluate(X[i])
            err += (Y[i] - y) ** 2
        err /= len(Y)
        return err

    def err_rate(self, X, Y):
        assert self.type == 'c'
        err = 0
        for i in range(len(Y)):
            y = self.evaluate(X[i])
            err += int(Y[i] != (y > 0.5))
        err /= len(Y)
        return err

    def size(self):
        return self.root.size()

    def print_tree(self):
        self.root.print_node(1)

    class Node:
        def __init__(self):
            self.isLeaf = True
            self.parent = None
            self.left = None
            self.right = None

        def learn_params(self, X, Y, V, R, alpha, tree):
            u = 0.1  # momentum weight
            eps = 0.00001

            ix = np.arange(len(Y))

            dw = np.zeros(X.shape[1])  # gradient of w
            dwp = np.zeros(X.shape[1])  # previous gradient of w

            dw10, dw20, dw0 = 0, 0, 0  # grads of w0
            dw10p, dw20p, dw0p = 0, 0, 0  # previous grads of w0

            for e in range(MAXEPOCH):
                np.random.shuffle(ix)

                for i in range(len(X)):
                    j = ix[i]
                    x = X[j]
                    r = Y[j]
                    y = tree.evaluate(x)
                    d = y - r

                    t = alpha * d
                    m = self
                    p = None

                    # compute negative gradient
                    while m.parent is not None:
                        p = m.parent
                        if m.isLeft:
                            t *= p.v
                        else:
                            t *= (1 - p.v)
                        m = m.parent

                    dw = (-t * (self.left.y - self.right.y) * (self.v) * (1 - self.v)) * x
                    dw0 = -t * (self.left.y - self.right.y) * (self.v) * (1 - self.v)
                    dw10 = -t * (self.v)
                    dw20 = -t * (1 - self.v)

                    # update params (params -= alpha*gradient)
                    self.w += dw + u * dwp
                    self.w0 += dw0 + u * dw0p
                    self.left.w0 += dw10 + u * dw10p
                    self.right.w0 += dw20 + u * dw20p

                    # update previous grads
                    dwp = dw.copy()
                    dw0p = dw0
                    dw10p = dw10
                    dw20p = dw20

                    # update y values of the nodes in the path
                    self.update_y(x)

        def update_y(self, x):
            if self.isLeaf:
                self.y = np.dot(x, self.w) + self.w0
            else:
                if x[self.d] < self.v:
                    self.left.update_y(x)
                    self.y = self.left.y
                else:
                    self.right.update_y(x)
                    self.y = self.right.y

        def evaluate(self, x):
            if self.isLeaf:
                return np.dot(x, self.w) + self.w0
            else:
                if x[self.d] < self.v:
                    return self.left.evaluate(x)
                else:
                    return self.right.evaluate(x)

        def splitNode(self, X, Y, V, R, tree):
            if len(Y) == 0:
                return

            if len(Y) == 1 or np.all(Y == Y[0]):
                return

            if np.all(X == X[0]):
                return

            assert len(V) == len(R) == X.shape[1]

            best_d, best_v, min_R = 0, 0, float('inf')
            for d in range(X.shape[1]):
                s = sorted(zip(X[:, d], Y))
                Xs, Ys = zip(*s)
                for i in range(1, len(Y)):
                    if Xs[i] > Xs[i - 1]:
                        Rl, Rr = R[d][:i], R[d][i:]
                        if len(Rl) == 0 or len(Rr) == 0:
                            continue
                        v = Xs[i]
                        Dr = self.DR(Y, Rl, Rr)
                        R = Dr[0]
                        if R < min_R:
                            min_R = R
                            best_d, best_v = d, v

            if min_R == float('inf'):
                return

            self.isLeaf = False
            self.d = best_d
            self.v = best_v

            xl, xr, yl, yr = [], [], [], []
            for i in range(len(Y)):
                if X[i, self.d] < self.v:
                    xl.append(X[i])
                    yl.append(Y[i])
                else:
                    xr.append(X[i])
                    yr.append(Y[i])

            xl, yl, xr, yr = np.array(xl), np.array(yl), np.array(xr), np.array(yr)

            assert len(xl) > 0 and len(xr) > 0

            self.left = tree.Node()
            self.left.parent = self
            self.left.splitNode(xl, yl, V, R, tree)

            self.right = tree.Node()
            self.right.parent = self
            self.right.splitNode(xr, yr, V, R, tree)

            self.update_y(X[0])

        def DR(self, Y, Rl, Rr):
            min_R = float('inf')
            best_c = None

            for c in np.unique(Y):
                il = np.where(Y[Rl] == c)[0]
                ir = np.where(Y[Rr] == c)[0]

                Rl_c = Rl[il]
                Rr_c = Rr[ir]

                R = (len(Rl_c) + len(Rr_c)) * self.G(Rl_c, Rr_c)
                if R < min_R:
                    min_R = R
                    best_c = c

            return min_R, best_c

        def G(self, Rl, Rr):
            if len(Rl) == 0 or len(Rr) == 0:
                return 1

            pl = np.mean(Rl)
            pr = np.mean(Rr)

            return len(Rl) * len(Rr) * (pl - pr) ** 2

        def size(self):
            if self.isLeaf:
                return 1
            else:
                return 1 + self.left.size() + self.right.size()

        def print_node(self, depth):
            if self.isLeaf:
                print(' ' * depth, 'Leaf:', self.y)
            else:
                print(' ' * depth, 'Branch:', self.d, self.v)
                self.left.print_node(depth + 1)
                self.right.print_node(depth + 1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
