import numpy as np  # for numerical computation (as the computing engine of pypop7)


# Rosenbrock (one notorious test function from the optimization community)
def rosenbrock(x):
    return 100.0 * np.sum(np.square(x[1:] - np.square(x[:-1]))) + np.sum(
        np.square(x[:-1] - 1.0)
    )


# to define the fitness function to be *minimized* and its problem settings
ndim_problem = 1000
problem = {
    "fitness_function": rosenbrock,
    "ndim_problem": ndim_problem,  # dimension
    "lower_boundary": -5.0 * np.ones((ndim_problem,)),  # lower search boundary
    "upper_boundary": 5.0 * np.ones((ndim_problem,)),
}  # upper search boundary

options = {
    "max_function_evaluations": 5000,  # to set optimizer options
    "n_individuals": 200,
}

from pypop7.optimizers.de.cde import CDE

de = CDE(problem, options)
results = de.optimize()
print(results)
