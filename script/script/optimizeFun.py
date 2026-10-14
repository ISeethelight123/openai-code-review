import numpy as np
from scipy.optimize import linprog
import time
import pandas as pd
import cvxpy as cp
from cvxopt import matrix, solvers
import clarabel
'''
linprog求解线性规划
'''

path = "D:/data/linprog/"
f = pd.read_csv(path + "f.csv", header=None)
f = np.array(f[0].tolist())
A = pd.read_csv(path + "A.csv",header=None).to_numpy()
b = pd.read_csv(path + "b.csv", header=None)
b = np.array(b[0].tolist())


# 等式约束
Aeq = np.ones((1, f.size))
beq = np.array([1])

# 记录开始时间
start_time = time.time()

# 使用线性规划库求解问题
result = linprog(f, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=[-10, 10])

# 记录结束时间
end_time = time.time()
execution_time = (end_time - start_time) * 1000  # 转换为毫秒
# 输出结果
print("Optimal value:", result.fun)
#print("Optimal solution:", result.x)
print("linprog求解时间: {} 毫秒".format(execution_time))

'''
cvxopt求解二次规划
'''

path = "D:/data/data_socp/"
f = pd.read_csv(path + "f.csv")
C = np.array(f["f"].tolist())
A = pd.read_csv(path + "A_ub.csv").to_numpy()
b = np.array(
    [0.025876723, 0.092515275, 0.035133942, 0.053184884, 0.067410565, 0.009709433, 0.04668745, 0.00636804, 0.022258664,
     0.11027537, 0.018488302, 0.027417204, 0.028585, 0.017228214, 0.008055527, 0.015727843, 0.026132369, 0.013646113,
     0.066000808,
     0.043606587, 0.048325258, 0.033868626, 0.010790603, 0.017737391, 0.03252374, 0.039329965, 0.040665779, 0.010868773,
     0.006819891,
     0.015879314, 0.008882335, -0.025876723, -0.092515275, -0.035133942, -0.053184884, -0.067410565, -0.009709433,
     -0.04668745,
     -0.00636804, -0.022258664, -0.110275379, -0.018488302, -0.027417204, -0.028585, -0.017228214, -0.008055527,
     -0.015727843, -0.026132369,
     -0.013646113, -0.066000808, -0.043606587, -0.048325258, -0.033868626, -0.010790603, -0.017737391, -0.03252374,
     -0.039329965, -0.040665779,
     -0.010868773, -0.006819891, -0.015879314, -0.008882335])

Aeq = np.ones((1, len(C)))
beq = [1]
H = np.eye(f.size)

x = cp.Variable(f.size)
prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, H) + C.T @ x), [A @ x <= b, Aeq @ x == beq])

start_time = time.time()
prob.solve()
end_time = time.time()

execution_time = (end_time - start_time) * 1000  # 转换为毫秒

# 打印结果
print("quad求解时间: {} 毫秒".format(execution_time))
print("\n最优值为", prob.value)
#print(x.value)

'''
cvxopt求解qclp
'''

import pandas as pd
import numpy as np
import cvxpy as cp
from cvxopt import matrix, solvers
import time
import clarabel

path = "D:/data/qclp/"

r = pd.read_csv(path + "r.csv", header=None)
r = np.array(r[0].tolist())
V = pd.read_csv(path + "V.csv", header=None).to_numpy()
k = 10000
A = np.eye(r.size)

b = pd.read_csv(path + "b.csv", header=None)
b = np.array(b[0].tolist())
Aeq = np.ones((1, len(r)))
beq = 7.3836

x = cp.Variable(r.size)
prob1 = cp.Problem(cp.Minimize(r.T @ x), [cp.quad_form(x, V) <= k, A @ x <= b, Aeq @ x == beq])

start_time = time.time()
prob1.solve(solver='CLARABEL', verbose=True)
end_time = time.time()

execution_time = (end_time - start_time) * 1000  # 转换为毫秒

# 打印结果
print("qclp求解时间: {} 毫秒".format(execution_time))
print("\n最优值为", prob1.value)
#print(x.value)