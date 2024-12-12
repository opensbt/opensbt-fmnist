import pymoo

from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

from fmnist.critical_fmnist import CriticalFMNIST
from opensbt.evaluation.fitness import *
from opensbt.experiment.search_configuration import DefaultSearchConfiguration
from opensbt.experiment.experiment import *
from opensbt.algorithm.algorithm import *
from opensbt.evaluation.critical import *
from opensbt.config import *
from opensbt.config import RESULTS_FOLDER as results_folder
from opensbt.algorithm.nsga2d_optimizer import NSGAIIDOptimizer

from fmnist.fmnist_problem import FMNISTProblem
from mnist.fitness_mnist import *
from mnist.config import EXPECTED_LABEL
from mnist.utils_mnist import get_number_verts

from fmnist import fmnist_simulation
from fmnist.fmnist_simulation import FMnistSimulator
from fmnist.fmnist_simulation import get_seeds_class
from fmnist.operator_fmnist import FMnistSamplingValid

import logging as log

""" FMNIST Testing with single seed mutation

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

# we select the first seed of images with corresponding label
seed = fmnist_simulation.get_seeds_class(EXPECTED_LABEL)[0]

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

''' MNIST Problem with single seed'''
config = DefaultSearchConfiguration()
config.population_size = 5
config.n_generations =  5
config.operators["init"] = FMnistSamplingValid

###########################
# 3 D Problem
######################

fmnistproblem = FMNISTProblem(
                        problem_name=f"FMNIST_3D",
                        xl=[lb, lb, 0],
                        xu=[ub, ub, ub_vert],
                        simulation_variables=[
                            "mut_extent_1",
                            "mut_extent_2",
                            "vertex_control"
                        ],
                        simulate_function=FMnistSimulator.simulate,
                        fitness_function=FitnessMNIST(diversify=True),
                        critical_function=CriticalFMNIST(),
                        expected_label=5,
                        min_saturation=0.1,
                        seed=seed
                        )
##############
# 6 D Problem
##############

# fmnistproblem = FMNISTProblem(
#                         problem_name=f"FMNIST_6D",
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
#                         simulate_function=FMnistSimulator.simulate,
#                         fitness_function=FitnessMNIST(),
#                         critical_function=CriticalFMNIST(),
#                         expected_label=EXPECTED_LABEL,
#                         min_saturation=0.1,
#                         max_seed_distance=4,
#                         seed=seed
#                         )

fmnistproblem.problem_name = fmnistproblem.problem_name+ "_NSGA-II-DJ" + f"_D{seed}" 
optimizer = NSGAIIDOptimizer(
    problem=fmnistproblem,
    config=config)

res = optimizer.run()

res.write_results(results_folder=results_folder, params = optimizer.parameters)

log.info("====== Algorithm search time: " + str("%.2f" % res.exec_time) + " sec")