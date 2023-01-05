import os
from evaluation.fitness import *
from evaluation import critical
from problem.adas_problem import ADASProblem
from problem.pymoo_test_problem import PymooTestProblem
from experiment.experiment import *
from simulation.simulator import SimulationType
from algorithm.algorithm import *
from evaluation.critical import *

'''
EXAMPLE CARLA SIMULATOR
ego speed is in km/h
'''

def getExp1() -> Experiment:
    from simulation.carla_simulation import CarlaSimulator

    problem = ADASProblem(
                          scenario_path=os.getcwd() + "/scenarios/PedestrianCrossing.xosc",
                          xl=[0.5, 1, 0],
                          xu=[3, 80, 60],
                          simulation_variables=[
                              "PedestrianSpeed",
                              "FinalHostSpeed",
                              "PedestrianEgoDistanceStartWalk"],
                          fitness_function=FitnessAdaptedDistanceSpeed(),
                          simulate_function=CarlaSimulator.simulate_batch,
                          critical_function=CriticalAdasFrontCollisions(),
                          simulation_time=10,
                          sampling_time=100,
                          problem_name="PedestrianCrossingStartWalk",
                          approx_eval_time=10,
                          do_visualize = False
                          )
    config = DefaultSearchConfiguration()
    config.max_tree_iterations = 3  # in paper 5-7
    # config.maximal_execution_time = None # limit the total time, to control the number of tree iterations
    experiment = Experiment(problem=problem,
                            algorithm=AlgorithmType.NSGAII,
                            search_configuration=config,
                            simulation_type=SimulationType.CARLA)
    return experiment

'''
    BNH Problem

    Pareto solutions:
    x∗1=x∗2∈[0,3]  and x∗1∈[3,5], x∗2=3
'''

def getExp2() -> Experiment:
    problem = PymooTestProblem(
        'BNH',
        critical_function=CriticalBnhDivided())

    config = DefaultSearchConfiguration()
    config.maximal_execution_time = "00:00:01"
    config.max_tree_iterations = 4
    experiment = Experiment(problem=problem,
                            algorithm=AlgorithmType.NSGAII,
                            search_configuration=config,
                            simulation_type=SimulationType.NONE)

    return experiment


experiment_switcher = {
    1: getExp1,
    2: getExp2,
}