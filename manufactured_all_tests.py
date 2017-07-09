from manufactured import *

F = 0.03; k = 0.06
time_step = "0.1"

end_time = "2.7"
time_step = "0.1"
degree = 1
output = "test/regular.pvd"

grayScottSolver( F, k, degree, end_time=end_time, time_step=time_step,
                 save_solution = True, output = output, mesh_size = 32, domain_size = 1.0)

end_time = "2.7"
time_step = "0.1"
degree = 2
output = "test/degree2.pvd"

grayScottSolver( F, k, degree, end_time=end_time, time_step=time_step,
                 save_solution = True, output = output, mesh_size = 32, domain_size = 1.0)

end_time = "2.7"
time_step = "0.1"
degree = 3
output = "test/degree3.pvd"

grayScottSolver( F, k, degree, end_time=end_time, time_step=time_step,
                 save_solution = True, output = output, mesh_size = 32, domain_size = 1.0)

end_time = "2.6"
time_step = "0.01"
degree = 1
output = "test/small_time_step.pvd"

grayScottSolver( F, k, degree, end_time=end_time, time_step=time_step,
                     save_solution = True, output = output, mesh_size = 32, domain_size = 1.0)

end_time = "2.6"
time_step = "0.001"
degree = 1
output = "test/extra_small_time_step.pvd"

grayScottSolver( F, k, degree, end_time=end_time, time_step=time_step,
                     save_solution = True, output = output, mesh_size = 32, domain_size = 1.0)
