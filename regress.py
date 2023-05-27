import numpy as np
from SoftTree import SoftTree

def read_from_file(filename):
    X = []
    Y = []

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip().split()
            row = [float(val) for val in line[:-1]]
            Y.append(float(line[-1]))
            X.append(row)

    return np.array(X), np.array(Y)

def normalize(X, V, U):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=1)

    X = (X - mean) / std
    V = (V - mean) / std
    U = (U - mean) / std

    return X, V, U

def main(filename, fold):
    m = int(fold / 2) + 1
    n = fold % 2 + 1

    dataset = f"data/{filename}/{filename}"

    train_filename = f"{dataset}-train-{m}-{n}.txt"
    X, Y = read_from_file(train_filename)

    validation_filename = f"{dataset}-validation-{m}-{n}.txt"
    V, R = read_from_file(validation_filename)

    test_filename = f"{dataset}-test.txt"
    U, T = read_from_file(test_filename)

    X, V, U = normalize(X, V, U)

    st = SoftTree(X, Y, V, R)

    mse_train = st.meanSqErr(X, Y)
    mse_validation = st.meanSqErr(V, R)
    mse_test = st.meanSqErr(U, T)

    print("SRT:")
    print(f"n: {st.size()}\ttra: {mse_train}\tval: {mse_validation}\ttst: {mse_test}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python regress.py <filename> <fold>")
        sys.exit(1)

    filename = sys.argv[1]
    fold = int(sys.argv[2])

    main(filename, fold)
