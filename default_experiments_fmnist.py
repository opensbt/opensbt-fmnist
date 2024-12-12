import os
from opensbt.evaluation.fitness import *
from opensbt.algorithm.algorithm import AlgorithmType
from opensbt.experiment.search_configuration import DefaultSearchConfiguration
from opensbt.experiment.experiment import *
from opensbt.algorithm.algorithm import *
from opensbt.evaluation.critical import *
from fmnist.fmnist_problem import *
from mnist.fitness_mnist import *
from mnist.utils_mnist import get_number_verts
from fmnist.operator_fmnist import FMnistSamplingValid
import copy
from opensbt.config import *
from fmnist.fmnist_simulation import FMnistSimulator, get_seeds_class
from opensbt.experiment.experiment_store import experiments_store
from fmnist.critical_fmnist import CriticalFMNIST
from mnist.config import EXPECTED_LABEL

""" FMNIST Problem with single seed

Fashion mnist classes

# 0 T-shirt/top
# 1 Trouser
# 2 Pullover
# 3 Dress
# 4 Coat
# 5 Sandal
# 6 Shirt
# 7 Sneaker
# 8 Bag
# 9 Ankle boot

"""

config = DefaultSearchConfiguration()
config.population_size = 2
config.n_generations =  2
config.operators["init"] = FMnistSamplingValid

# we select the first seed of images with corresponding label
seed = get_seeds_class(EXPECTED_LABEL)[2]

# control the extent for the mutation (num pixels)
lb = -3
ub = +3

digit = fmnist_simulation.generate_and_evaluate_digit(seed)

print("Class predicted:", digit.predicted_label)

# check if classification for seed is correct, otherwise mutation does not makes sense
# exit if seed gets already misclassified
if digit.predicted_label != EXPECTED_LABEL:
    log.info("Seed gets misclassified. Skipping seed and stopping execution...")
    sys.exit(0)

vertex_num = get_number_verts(digit)
ub_vert = vertex_num -1 

# config.operators["mut"] = MnistMutation
# config.operators["cx"] = MyNoCrossover
# config.operators["dup"] = MnistDuplicateElimination
config.operators["init"] = FMnistSamplingValid

# Select between 3D and 6D problem

# mnistproblem = MNISTProblem(
#                         problem_name=f"MNIST_3D",
#                         xl=[lb, lb, 0],
#                         xu=[ub, ub, ub_vert],
#                         simulation_variables=[
#                             "mut_extent_1",
#                             "mut_extent_2",
#                             "vertex_control"
#                         ],
#                         simulate_function=FMnistSimulator.simulate,
#                         fitness_function=FitnessMNIST(),
#                         critical_function=CriticalMNISTConf_05(),
#                         expected_label=5,
#                         min_saturation=0.1,
#                         seed=seed
#                         )

mnistproblem = FMNISTProblem(
                        problem_name=f"FMNIST_6D",
                        xl=[lb, lb, lb, lb,  0, 0],
                        xu=[ub, ub, ub, ub,  ub_vert, ub_vert],
                        simulation_variables=[
                            "mut_extent_1",
                            "mut_extent_2",
                            "mut_extent_3",
                            "mut_extent_4",
                            "vertex_control",
                            "vertex_start"
                        ],
                        simulate_function=FMnistSimulator.simulate,
                        fitness_function=FitnessMNIST(),
                        critical_function=CriticalFMNIST(),
                        expected_label=EXPECTED_LABEL,
                        min_saturation=0.1,
                        max_seed_distance=4,
                        seed=seed
                        )
#############################################
''' NSGA-II with optimizing diversity using repopulation operator and smart archive - ARCHIVE THS 0'''
def getExp201() -> Experiment:
    problem = copy.deepcopy(mnistproblem)
    problem.set_fitness_function(FitnessMNIST(diversify=True))
    problem.critical_function=CriticalFMNIST()
    problem.problem_name = problem.problem_name+ "_NSGA-II" + f"_D{seed}" 
    config.archive_threshold = 5
    config.n_repopulate_max = 0.5
    config.mode_processing = 2

    experiment = Experiment(name="201",
                            problem=problem,
                            algorithm=AlgorithmType.NSGAII,
                            search_configuration=config)
    return experiment

experiments_store.register(getExp201())

''' Grid sampling '''
def getExp701() -> Experiment:
    problem = copy.deepcopy(mnistproblem)
    problem.set_fitness_function(FitnessMNIST(diversify=False))
    problem.critical_function=CriticalFMNIST()
    problem.problem_name = problem.problem_name + "_GS" + f"_D{seed}" 

    experiment = Experiment(name="701",
                            problem=problem,
                            algorithm=AlgorithmType.PS_GRID,
                            search_configuration=config)
    return experiment
experiments_store.register(getExp701())

''' NSGA-II-DT '''
def getExp801() -> Experiment:
    problem = copy.deepcopy(mnistproblem)
    problem.set_fitness_function(FitnessMNIST(diversify=False))
    problem.critical_function=CriticalFMNIST()
    problem.problem_name = problem.problem_name + "_NSGA-II-DT" +  f"_D{seed}" 
    config.inner_num_gen = 5
    
    experiment = Experiment(name="801",
                            problem=problem,
                            algorithm=AlgorithmType.NSGAII_DT,
                            search_configuration=config)
    return experiment
experiments_store.register(getExp801())