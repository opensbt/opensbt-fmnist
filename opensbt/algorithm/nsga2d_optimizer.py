
from opensbt.utils.archive import MemoryArchive
import pymoo

from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import SimulationResult
pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

from opensbt.algorithm.nsga2d.archive import ProcessingMode
from opensbt.utils.operators import select_operator
from opensbt.algorithm.nsga2d.nsga2d import NSGA2D
from opensbt.algorithm.optimizer import Optimizer
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from algorithm.classification.decision_tree.decision_tree import *
from experiment.search_configuration import SearchConfiguration
import logging as log
from model_ga.result import *
from config import *

class NSGAIIDOptimizer(Optimizer):

    """ Extension of NSGA-II algorithm to include concepts from Novelty Search.
        Holdsan archive to measure during search diversity to already found individuals and uses repopulation.
        Implementation is from the paper:

        Sorokin, L., Safin, D. & Nejati, S. Can search-based testing with pareto optimization effectively 
        cover failure-revealing test inputs?. Empir Software Eng 30, 26 (2025). 
        https://doi.org/10.1007/s10664-024-10564-3

        Original idea is taken from the DeepJanus approach.
    """
    
    algorithm_name =  "NSGA-II-D"

    def __init__(self,
                problem: Problem,
                config: SearchConfiguration):

        self.config = config
        self.problem = problem
        self.res = None

        if self.config.prob_mutation is None:
            self.config.prob_mutation = 1 / problem.n_var

        self.algorithm = NSGA2D(
            pop_size=config.population_size,
            n_offsprings=config.num_offsprings,
            sampling=select_operator("init", config),
            crossover=select_operator("cx", config),
            mutation=select_operator("mut", config),
            eliminate_duplicates=select_operator("dup", config),
            n_repopulate_max=config.n_repopulate_max,
            archive_threshold=config.archive_threshold,
            mode_processing=ProcessingMode(config.mode_processing),
            archive= MemoryArchive())

        ''' Prioritize max search time over set maximal number of generations'''
        if config.maximal_execution_time is not None:
            self.termination = get_termination("time", config.maximal_execution_time)
        else:
            self.termination = get_termination("n_gen", config.n_generations)

        self.save_history = True
        self.parameters = {
            "Population size" : str(config.population_size),
            "Number of generations" : str(config.n_generations),
            "Number of offsprings": str(config.num_offsprings),
            "Crossover probability" : str(config.prob_crossover),
            "Crossover eta" : str(config.eta_crossover),
            "Mutation probability" : str(config.prob_mutation),
            "Mutation eta" : str(config.eta_mutation),
            "Seed" : str(config.seed),
            "N_repopulate_max" : config.n_repopulate_max,
            "Archive_threshold" : config.archive_threshold,
            "Mode_processing" : ProcessingMode(config.mode_processing)
        }

        log.info(f"Initialized algorithm with config: {config.__dict__}")

    def run(self) -> SimulationResult:
        self.res = minimize(self.problem,
                    self.algorithm,
                    self.termination,
                    save_history=self.save_history,
                    verbose=True)

        return self.res

# if __name__ == "__main__":        
    
#     # problem = PymooTestProblem(
#     #     'BNH', critical_function=CriticalBnhDivided())
#     # config = DefaultSearchConfiguration()

#     # class CriticalMW1(Critical):
#     #     def eval(self, vector_fitness: List[float], simout: SimulationOutput = None):
#     #         if vector_fitness[0] <= 0.8 and vector_fitness[0] >= 0.2 and \
#     #            vector_fitness[1] <= 0.8 and vector_fitness[0] >= 0.2:
#     #             return True
#     #         else:
#     #             return False
            
#     problem = PymooTestProblem(
#         'bnh', critical_function=CriticalBnhDivided())
#     config = DefaultSearchConfiguration()

#     config.population_size = 100
#     config.n_generations = 10
#     config.prob_mutation = 0.5
#     config.n_func_evals_lim = 20
#     config.n_repopulate_max = 10

#     optimizer = NSGAIID_SIM(problem,config)
#     optimizer.run()
#     optimizer.write_results(
#         ref_point_hv=np.asarray([200,200]), 
#         ideal = np.asarray([0,0]), 
#         nadir = np.asarray([200,200]), 
#         results_folder = RESULTS_FOLDER)