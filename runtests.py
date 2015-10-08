import optspace
import numpy as np
import numpy.random as npr

U = npr.randn(1000, 10)
V = npr.randn(10, 1000)
mat = np.dot(U, V)

smat = []
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        if npr.rand() < 0.1:
            smat.append((i, j, mat[i][j]))

(X, S, Y) = optspace.optspace(smat, rank_n = 10,
    num_iter = 10000,
    tol = 1e-4,
    verbosity = 1,
    outfile = ""
)

[X, S, Y] = map(np.matrix, [X, S, Y])

print 'X', X
print 'Y', Y
print 'S', S
print 'X * S * Y^T', X * S * Y.T
print 'mat', mat
print 'rmse', np.sqrt(np.sum(np.power(X * S * Y.T - mat, 2)) / X.shape[0] / Y.shape[0])
