from manufactured import *
from scipy.io import savemat
import numpy as np

F = 0.03
k = 0.06

Ns = [1024, 512, 256, 128, 64, 32, 16]
Ts = ["0.001", "0.002", "0.005", "0.01", "0.02", "0.05", "0.1"]
end_time = "1.0"

degree = 3

errors_space = np.zeros((len(Ns), 2))
errors_time = np.zeros((len(Ts), 2))

# The errors refining the mesh size
for i, mesh_size in enumerate(Ns):
    l2_err, infty_err = grayScottSolver( F, k, degree, end_time = end_time, time_step = "0.05",
                                         mesh_size = mesh_size, domain_size = 1.0)
    errors_space[i,:] = l2_err, infty_err

# The errors refining the time step
for i, time_step in enumerate(Ts):
    l2_err, infty_err = grayScottSolver( F, k, degree, end_time = end_time, time_step = time_step,
                                         mesh_size = 2**10, domain_size = 1.0)
    errors_time[i,:] = l2_err, infty_err

savemat("manufactured_degree3.mat", mdict = {"errors_space": errors_space, "errors_time": errors_time})
