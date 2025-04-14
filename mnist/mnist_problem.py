from dataclasses import dataclass
from typing import Dict
import numpy as np
from opensbt.evaluation.critical import Critical
from opensbt.evaluation.fitness import *
import logging as log
from os.path import join
from pathlib import Path

from mnist import utils_mnist
from mnist.utils import string_utils
from mnist import mnist_simulation
from typing import List
from pymoo.core.problem import Problem

@dataclass
class MNISTProblem(Problem):

    def __init__(self,
                 xl: List[float], 
                 xu: List[float], 
                 simulate_function, # the MnistSimulator
                 fitness_function: Fitness, 
                 critical_function: Critical, 
                 simulation_variables: List[float], 
                 design_names: List[str] = None, 
                 objective_names: List[str] = None, 
                 problem_name: str = None, 
                 other_parameters: Dict = None,
                 expected_label: int = None,
                 max_seed_distance: float = 3,
                 min_saturation: float = 0,
                 seed: int = None):

        super().__init__(n_var=len(xl),
                         n_obj=len(fitness_function.name),
                         xl=xl,
                         xu=xu)
        
        assert simulate_function is not None
        assert xl is not None
        assert xu is not None
        assert np.equal(len(xl), len(xu))
        assert np.less_equal(xl, xu).all()
        assert expected_label is not None

        self.set_fitness_function(fitness_function, objective_names)

        self.critical_function = critical_function
        self.simulation_variables = simulation_variables
        self.expected_label = expected_label
        self.max_seed_distance = max_seed_distance
        self.min_saturation = min_saturation
        self.problem_name = problem_name
        self.simulate_function = simulate_function

        print(f"problem config: {self.__dict__}")

        self.set_seed(seed)

        if design_names is not None:
            self.design_names = design_names
        else:
            self.design_names = simulation_variables

        self.other_parameters = other_parameters

        self.counter = 0

    def set_seed(self, seed):
        log.info(f"Seed of MNISTProblem set to: {seed}")

        digit = mnist_simulation.generate_and_evaluate_digit(seed)
        segment_num = utils_mnist.get_number_segments(digit)
        vertex_num = utils_mnist.get_number_verts(digit)
      
        # adapt the search space as number of  vertices is required in space specification
        s_size = len(self.xu)
        if s_size == 3:
            self.xu[s_size-1] = vertex_num - 1
        if s_size == 8:
            self.xu[s_size-1] = vertex_num - 1
            self.xu[s_size-2] = vertex_num - 1

        self.seed = seed
        self.vertex_num = vertex_num
        self.segment_num = segment_num
        self.seed_digits = [
               digit
        ]
        # hack: replace name so that seed is in name
        self.problem_name = string_utils.replace_last_number_with_new_number(self.problem_name, seed)
                
    def set_fitness_function(self, fitness_function, objective_names = None):
        assert fitness_function is not None
        assert len(fitness_function.min_or_max) == len(fitness_function.name)
        
        self.n_obj=len(fitness_function.name)

        self.fitness_function = fitness_function

        if objective_names is not None:
            self.objective_names = objective_names
        else:
            self.objective_names = fitness_function.name

        self.signs = []
        for value in self.fitness_function.min_or_max:
            if value == 'max':
                self.signs.append(-1)
            elif value == 'min':
                self.signs.append(1)
            else:
                raise ValueError(
                    "Error: The optimization property " + str(value) + " is not supported.")

        
    def _evaluate(self, x, out, *args, **kwargs):
        vector_list = []
        label_list = []
        digits = []
        self.counter = self.counter + 1

        for i, ind in enumerate(x):
            kwargs["individual"] = ind
            kwargs["problem"] = self

            simout: SimulationOutput = self.simulate_function([ind],
                                            self.simulation_variables, 
                                            scenario_path = "No_Path_Required", 
                                            sim_time= -1,
                                            time_step= -1,
                                            **kwargs)[0]
            vector_fitness = np.asarray(
                self.fitness_function.eval(simout=simout,
                                           **kwargs)
            )

            # set structure
            signed_fitness = np.asarray(self.signs) * np.array(vector_fitness)
            vector_list.append(signed_fitness)
            label_list.append(self.critical_function.eval(signed_fitness,
                                                          simout=simout))
            digits.append(simout.otherParams["DIG"])

        out["F"] = np.vstack(vector_list)
        out["CB"] = label_list
        out["DIG"] = digits
        
    def is_simulation(self):
        return False

  
  