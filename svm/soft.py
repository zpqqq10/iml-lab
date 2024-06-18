import pickle
from icecream import ic 
import numpy as np
from cvxopt import matrix, solvers
from libsvm.svmutil import *


# https://blog.csdn.net/zyx_bx/article/details/118021462
# https://xavierbourretsicotte.github.io/SVM_implementation.html
# soft margin
class SVM:
    def __init__(self):
        # cost, costant C
        self.C = 1.0
        # max iterations, same as libsvm
        self.iter = 10000000
    
    def read_problem(self, y, x):
        # number of data points
        N = len(y)
        # N x 1
        self.labels = np.array(y).reshape(-1, 1) * 1.
        # determine dimensions
        dims = set()
        for row in x:
            # not all data points have all dimensions
            dims = dims.union(set(row.keys()))
        dims = len(dims)
        # plus one for ionosphere, since the feature dim is actually 1-34 without 2
        dims += 1
        data = np.zeros((N, dims))
        # key begins from 1
        for i in range(N):
            for k, v in x[i].items():
                data[i, k - 1] = v
        # N x d
        self.data = data
        return self.labels, self.data
    
    def set_parameters(self, cost = 1.0, iter = 10000000):
        self.C = cost
        self.iter = iter
    
    def train(self, solve_dual = True):
        solvers.options['show_progress'] = False
        # https://cvxopt.org/userguide/coneprog.html?highlight=qp#s-external
        # self.C = self.C / len(self.labels)
        solvers.options['maxiters'] = self.iter
        return self._solve_dual() if solve_dual else self._solve_primal()
    
    # minimize 1/2 ||w||^2 + C\sum\xi_i
    # s.t. y_i(w*x_i + b) >= 1 - \xi_i, \xi_i >= 0
    # x = [w, b, \xi] d + 1 + N
    # w = x * w_mask, b = x * b_mask, \xi = x * xi_mask
    # minimize 1/2 w^T @ w + C.T @ \xi
    # s.t. -Y * (wX + b) - \xi <= -1, -\xi <= 0
    def _solve_primal(self):
        N, dims = self.data.shape
        params = dims + 1 + N
        x = self.data
        y = self.labels
        # append 1 dimension, so w'X' = wX + b (w' = [w, b], X' = [X, 1])
        x = np.hstack((x, np.ones((N, 1))))

        w_mask_m = np.zeros((params, params))
        w_mask_m[:dims, :dims] = np.eye(dims)
        wb_mask_m = np.zeros((dims+1, params))
        wb_mask_m[:dims+1, :dims+1] = np.eye(dims+1)
        xi_mask_m = np.zeros((N, params))
        xi_mask_m[-N:, -N:] = np.eye(N)
        xi_mask_v = np.zeros((params, 1))
        xi_mask_v[-N:, 0] = 1
        
        P = matrix(w_mask_m)
        q = matrix(self.C * xi_mask_v)
        G = matrix(np.vstack((-y * x @ wb_mask_m - xi_mask_m, -xi_mask_m)))
        h = matrix(np.vstack((-np.ones((N, 1)), np.zeros((N, 1)))))
        # solve QP
        sol = solvers.qp(P, q, G, h)
        wb = np.array(sol['x'])[:dims+1]
        xi = np.array(sol['x'])[-N:]
        
        # calculate accuracy
        pred = np.sign(x @ wb)
        acc = np.mean(pred == y)
        mse = np.mean(np.square(pred - y))
        print('Solving primal problem')
        print(f'Accuracy: {acc} ({np.sum(pred == y)} / {len(y)}) MSE: {mse}')
        return pred, acc, wb
    
    
    # maximize \sum\alpha_i - 1/2 \sum\sum\alpha_i\alpha_jy_iy_j<x_i, x_j>
    # s.t. 0 <= \alpha_i <= C, \sum\alpha_iy_i = 0
    # that is, in matrix minimization form
    # matrix xy = Y * X
    # matrix H = xy @ xy.T
    # minimize 1/2\alpha^T @ H @ \alpha - 1.T @ \alpha
    # s.t. 0 <= \alpha <= C, Y^T @ \alpha = 0
    def _solve_dual(self):
        N, dims = self.data.shape
        x = self.data
        y = self.labels
        # matrix xy
        XY_m = y * x
        # matrix H
        mH = XY_m @ XY_m.T
        # define QP parameters of cvxopt
        P = matrix(mH)
        q = matrix(-np.ones((N, 1)))
        G = matrix(np.vstack((-np.eye(N), np.eye(N))))
        h = matrix(np.vstack((np.zeros((N, 1)), self.C * np.ones((N, 1)))))
        A = matrix(y.T)
        b = matrix(np.zeros(1))
        
        # solve QP
        sol = solvers.qp(P, q, G, h, A, b)
        alpha = np.array(sol['x'])
        # print(alpha[alpha > 1e-4])
        w = ((y * alpha).T @ x).reshape(-1, 1)
        # think alpha > 1e-4 as support vectors ( 0 < alpha <= C)
        index = (alpha > 1e-4).flatten()
        b = y[index] - x[index] @ w
        assert len(b) > 0, 'No support vectors found'
        print(f'Number of support vectors: {len(b)}')
        # average b
        b = np.mean(b)
        # b = b[0]
        
        # calculate accuracy
        pred = np.sign(x @ w + b)
        acc = np.mean(pred == y)
        mse = np.mean(np.square(pred - y))
        print('Solving dual problem')
        print(f'Accuracy: {acc} ({np.sum(pred == y)} / {len(y)}) MSE: {mse}')
        # print(np.where(pred != y))
        return pred, acc, np.vstack((w, b))

def read_pkl(path):
    with open(path, 'rb') as f:
        df = pickle.load(f)
    # convert labels
    df['diagnosis'] = df['diagnosis'].map({'M': -1., 'B': 1.})
    y = df['diagnosis'].tolist()
    
    # convert data
    data_cols = df.columns[2:]
    x = []
    index_counter = 1
    for index, row in df.iterrows():
        row_dict = {}
        for col in data_cols:
            row_dict[index_counter] = row[col]
            index_counter += 1
        # reset index_counter
        index_counter = 1
        x.append(row_dict)
    assert len(y) == len(x), 'Length of y and x must be the same'
    return y, x     

# read a txt in libsvm format
def read_txt(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    y = []
    x = []
    for line in lines:
        items = line.strip().split()
        y.append(float(items[0]))
        row_dict = {}
        for item in items[1:]:
            k, v = item.split(':')
            row_dict[int(k)] = float(v)
        x.append(row_dict)
    return y, x 

# print wb from different methods and compare
def compare_wb(primal_wb, dual_wb, lib_wb):
    print(f'dual wb: {dual_wb.T}')
    print(f'primal wb: {primal_wb.T}')
    print(f'libsvm wb: {lib_wb.T}')
    print('------------------------')
    comp = np.abs(dual_wb - primal_wb)
    print(f'wb between primal and dual: {comp.T}')
    comp = np.abs(dual_wb - lib_wb)
    print(f'wb between dual and libsvm: {comp.T}')
    print('------------------------')

if __name__ == '__main__':
    C = 8.0
    y, x = read_pkl('data.pkl')       # c = 8.0
    # y, x = read_txt('aus.txt')        # c = 2.0
    # y, x = read_txt('iono.txt')       # c = 8.0
    train_y, train_x = y[:400], x[:400]
    # train_y, train_x = y, x
    # define problem and parameter
    svm = SVM()
    maty, matx = svm.read_problem(train_y, train_x)
    svm.set_parameters(C)
    dual_prediction, dual_acc, dual_wb = svm.train(True)
    primal_prediction, primal_acc, primal_wb = svm.train(False)
    
    # define problem and parameter
    prob = svm_problem(train_y, train_x)
    param = svm_parameter(f'-t 0 -c {C} -q') 
    model = svm_train(prob, param)
    
    # retain w and b from libsvm
    sv_indices = np.array(model.get_sv_indices())
    print(f'Number of support vectors: {len(sv_indices)}')
    sv_coef = np.array(model.get_sv_coef())
    sv = matx[sv_indices - 1]
    ww = sv.T @ sv_coef
    bb = -model.rho[0]
    lib_wb = np.vstack((ww, bb))
    compare_wb(primal_wb=primal_wb, dual_wb=dual_wb, lib_wb=lib_wb)
    pred = np.sign(matx @ ww + bb)
    # print(np.mean(pred == maty))
    svm_save_model('model.model', model)
    p_label, p_acc, p_val = svm_predict(train_y, train_x, model)
    
    ic(p_acc)
    print('-------------------------------------')
    test_y, test_x = y[400:], x[400:]
    maty, matx = svm.read_problem(test_y, test_x)
    p_label, p_acc, p_val = svm_predict(test_y, test_x, model)
    ic(p_acc)
    pred = np.sign(matx @ dual_wb[:-1] + dual_wb[-1])
    mse = np.mean(np.square(pred - maty))
    print(f'dual test: {np.mean(pred == maty)} ({np.sum(pred == maty)} / {len(maty)})  MSE: {mse}')
    pred = np.sign(matx @ primal_wb[:-1] + primal_wb[-1])
    mse = np.mean(np.square(pred - maty))
    print(f'primal test: {np.mean(pred == maty)} ({np.sum(pred == maty)} / {len(maty)})  MSE: {mse}')
    
    