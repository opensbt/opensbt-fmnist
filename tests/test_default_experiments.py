import os
from pathlib import Path
import subprocess
from typing import Dict
import pymoo
import time

from examples.carla.carla_simulation import CarlaSimulator
from opensbt.experiment.experiment import Experiment
from opensbt.model_ga.individual import IndividualSimulated
from tests import test_base
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import SimulationResult
pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

from opensbt.algorithm.nsga2_optimizer import NsgaIIOptimizer
from opensbt.evaluation.critical import CriticalAdasDistanceVelocity
from opensbt.experiment.search_configuration import DefaultSearchConfiguration
from opensbt.evaluation.fitness import FitnessMinDistanceVelocityFrontOnly
from opensbt.problem.adas_problem import ADASProblem

from opensbt.experiment.experiment_store import experiments_store
from default_experiments import *
import logging as log

class TestDefaultExperiments():

    @staticmethod
    def test_dummy_experiments_no_args():
        """ Go over all predefined experiments and run them. Right now only experiments are tested that 
            do not require GPU access. I.e., CARLA based experiments are not tested. """
        
        store : Dict[str, Experiment] = experiments_store.get_store()
        
        assert len(store) > 0

        for name, exp in store.items():
            print(exp.problem)
            if exp.problem.is_simulation() and exp.problem.simulate_function == CarlaSimulator.simulate:
                continue
            log.info("Starting experiment:", name)
            if os.path.exists('.\\venv\\Scripts\\Activate'):
                venv_path = '.\\venv\\Scripts\\Activate &&'
                prefix = ['cmd', '/c']
            elif os.path.exists('.\\venv\\bin\\activate'):
                venv_path = '.\\venv\\bin\\activate &&'
                prefix = ["source"]
            else:
                venv_path = ""
                prefix = [""]

            result = subprocess.run( prefix + [f'{venv_path} python', 
                                    'run.py', "-e", name], 
                                    shell=True,
                                    capture_output=True, 
                                    text=True)
            log.info("Finished experiment")

            assert result.returncode == 0