import os
from opensbt.evaluation.fitness import *
from opensbt.evaluation import critical

from opensbt.algorithm.algorithm import AlgorithmType
from opensbt.experiment.search_configuration import DefaultSearchConfiguration
from opensbt.problem.adas_problem import ADASProblem
from opensbt.problem.pymoo_test_problem import PymooTestProblem
from opensbt.experiment.experiment import *
from opensbt.algorithm.algorithm import *
from opensbt.evaluation.critical import *
from mnist.mnist_problem import *
from mnist.fitness_mnist import *
from mnist.utils_mnist import get_number_segments, get_number_verts
from mnist.operator import MnistSamplingValid
import copy
from opensbt.config import *
from mnist import mnist_simulation
from mnist.mnist_simulation import MnistSimulator
from opensbt.experiment.experiment_store import experiments_store

''' MNIST Problem with single seed'''
config = DefaultSearchConfiguration()
config.population_size = 10
config.n_generations =  10
config.operators["init"] = MnistSamplingValid

seed = 120 #127 #52# 132 #129
#other possible seeds: 8, 15, 23, 45, 52, 53, 102, 120, 127, 129, 132, 152
lb = -8
ub = +8

digit = mnist_simulation.generate_digit(seed)
vertex_num = get_number_verts(digit)
ub_vert = vertex_num -1 

# config.operators["mut"] = MnistMutation
# config.operators["cx"] = MyNoCrossover
# config.operators["dup"] = MnistDuplicateElimination
config.operators["init"] = MnistSamplingValid

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
#                         expected_label=5,
#                         min_saturation=0.1,
#                         max_seed_distance=4,
#                         seed=seed
#                         )

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
#############################################
''' NSGA-II with optimizing diversity using repopulation operator and smart archive - ARCHIVE THS 0'''
def getExp1000() -> Experiment:
    problem = copy.deepcopy(mnistproblem)
    problem.set_fitness_function(FitnessMNIST(diversify=True))
    problem.critical_function=CriticalMNISTConf_05()
    problem.problem_name = problem.problem_name+ "_NSGA-II-DJ" + f"_D{seed}" 
    config.archive_threshold = 0
    config.n_repopulate_max = 0.1
    config.mode_processing = 2

    experiment = Experiment(name="1000",
                            problem=problem,
                            algorithm=AlgorithmType.NSGAII,
                            search_configuration=config)
    return experiment

experiments_store.register(getExp1000())