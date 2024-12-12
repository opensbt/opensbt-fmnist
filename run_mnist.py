
from opensbt.algorithm.nsga2_optimizer import NsgaIIOptimizer
from opensbt.evaluation.fitness import *
from opensbt.experiment.search_configuration import DefaultSearchConfiguration
from opensbt.experiment.experiment import *
from opensbt.algorithm.algorithm import *
from opensbt.evaluation.critical import *
from mnist.mnist_problem import *
from mnist.fitness_mnist import *
from mnist.utils_mnist import get_number_verts
from mnist.operator import MnistSamplingValid
from opensbt.config import *
from mnist import mnist_simulation
from mnist.mnist_simulation import MnistSimulator
from opensbt.config import RESULTS_FOLDER as results_folder

""" MNIST Testing with single seed mutation 

"""

config = DefaultSearchConfiguration()
config.population_size = 2
config.n_generations =  2

### pass here custom operators ####

# config.operators["mut"] = MnistMutation
# config.operators["cx"] = MyNoCrossover
# config.operators["dup"] = MnistDuplicateElimination
config.operators["init"] = MnistSamplingValid

seed = 120 #127 #52# 132 #129
#other possible seeds: 8, 15, 23, 45, 52, 53, 102, 120, 127, 129, 132, 152

lb = -8  # displacement bounds
ub = +8

digit = mnist_simulation.generate_digit(seed)
vertex_num = get_number_verts(digit)
ub_vert = vertex_num -1 

config.operators["init"] = MnistSamplingValid

mnistproblem = MNISTProblem(
                        problem_name=f"MNIST_3D",
                        xl=[lb, lb, 0],
                        xu=[ub, ub, ub_vert],
                        simulation_variables=[
                            "mut_extent_1",
                            "mut_extent_2",
                            "vertex_control"
                        ],
                        simulate_function=MnistSimulator.simulate,
                        fitness_function=FitnessMNIST(),
                        critical_function=CriticalMNISTConf_05(),
                        expected_label=5,
                        min_saturation=0.1,
                        seed=seed
                        )
##############
# 6 D Problem
##############

# mnistproblem = MNISTProblem(
#                         problem_name=f"MNIST_6D",
#                         xl=[lb, lb, lb, lb,  0, 0],
#                         xu=[ub, ub, ub, ub,  ub_vert, ub_vert],
#                         simulation_variables=[
#                             "mut_extent_1",
#                             "mut_extent_2",
#                             "mut_extent_3",
#                             "mut_extent_4",
#                             "vertex_control",
#                             "vertex_start"
#                         ],
#                         simulate_function=MnistSimulator.simulate,
#                         fitness_function=FitnessMNIST(),
#                         critical_function=CriticalMNISTConf_05(),
#                         expected_label=EXPECTED_LABEL,
#                         min_saturation=0.1,
#                         max_seed_distance=4,
#                         seed=seed
#                         )

mnistproblem.set_fitness_function(FitnessMNIST(diversify=True))
mnistproblem.critical_function=CriticalMNISTConf_05()
mnistproblem.problem_name = mnistproblem.problem_name+ "_NSGA-II-DJ" + f"_D{seed}" 

optimizer = NsgaIIOptimizer(
    problem=mnistproblem,
    config=config)

res = optimizer.run()

res.write_results(results_folder=results_folder, params = optimizer.parameters)

log.info("====== Algorithm search time: " + str("%.2f" % res.exec_time) + " sec")