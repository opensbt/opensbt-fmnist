from typing import List, Dict
from fmnist.fmnist_loader import fmnist_loader
from mnist import features, vectorization_tools
from fmnist import predictor
from mnist.digit_input import Digit
from mnist.config import EXPECTED_LABEL
from opensbt.simulation.simulator import Simulator, SimulationOutput
import json
from scipy.stats import entropy
import numpy as np
import logging as log
from mnist.mutations import *

class FMnistSimulator(Simulator):
    do_visualize = True
    sim_time = 2
    time_step = 0.01

    @staticmethod
    def simulate(list_individuals, 
                variable_names, 
                scenario_path: str, 
                sim_time: float, 
                time_step = 1,
                do_visualize = False,
                **kwargs) -> List[SimulationOutput]:
        try:
            results = []
            problem = kwargs["problem"]

            seed_digits = problem.seed_digits

            for ind in list_individuals:
                out = {}

                # apply mutations only to the same seed digit
                new_digit = seed_digits[0].clone()

                if len(variable_names) == 3:
                    # create digit
                    extent_1 = ind[0]
                    extent_2 = ind[1] 
                    vertex = round(ind[2])
                    
                    new_digit = apply_mutation_index(problem, new_digit, extent_1, extent_2, vertex)
                elif len(variable_names) == 6:
                    # create digit
                    extent_1 = ind[0]
                    extent_2 = ind[1] 
                    vertex_1 = round(ind[4])
                    
                    # create digit
                    extent_3 = ind[2]
                    extent_4 = ind[3] 
                    vertex_2 = round(ind[5])
                    
                    new_digit = apply_mutation_index_bi(problem,
                                    new_digit, 
                                    extent_1, 
                                    extent_2, 
                                    extent_3, 
                                    extent_4, 
                                    vertex_1, 
                                    vertex_2)

                assert(new_digit.seed == problem.seed_digits[0].seed)
                
                ##### Evalute fitness value of the classification ( = simulation) ########
                predicted_label, confidence = \
                        predictor.Predictor.predict(new_digit.purified, problem.expected_label)
                predictions = predictor.Predictor.predict_extended(new_digit.purified, problem.expected_label)

                ##### store info in digit ##########
                new_digit.predicted_label = predicted_label
                new_digit.confidence = confidence

                brightness = new_digit.brightness(min_saturation=problem.min_saturation)
                coverage = new_digit.coverage(min_saturation=problem.min_saturation)
                coverage_rel = new_digit.coverage(
                                            min_saturation=problem.min_saturation,
                                            relative = True
                )
                # calculate static and dynamic properties
                data = {}
                data["predicted_label"] = predicted_label
                data["confidence"] = confidence
                data["predictions"] = predictions
                data["expected_label"] = problem.expected_label
                # data["archive"] = archive # all digits found so far # TODO improve how we pass the archive
                data["digit"] = new_digit
                # data["distance_archive"] = distance
                data["coverage"] = coverage
                data["brightness"] = brightness
                data["move_distance"] = features.move_distance(new_digit)
                data["angle"] = features.angle_calc(new_digit)
                data["orientation"] = features.orientation_calc(new_digit, problem.min_saturation)
                data["entropy_signed"] = - entropy(pk=predictions) if np.argmax(predictions) != problem.expected_label else entropy(pk=predictions)
                # data["distance_test_input"] = distance_test_input
                data["coverage_rel"] = coverage_rel
                
                log.info("Individual evaluated and mutated digit created.")
                
                dict_simout = {
                        "simTime": 0.0,
                        "times": [],
                        "location": {},
                        "velocity": {},
                        "speed": {},
                        "acceleration": {},
                        "yaw": {},
                        "collisions": [],
                        "actors": {},
                        "otherParams": {}
                    }
                # create artifical simout
                simout = SimulationOutput.from_json(json.dumps(dict_simout))
              
                # fill 
                simout.otherParams["data"] = data
                simout.otherParams["DIG"] = new_digit

                results.append(simout)

        except Exception as e:
            raise e
        return results

def generate_digit(seed):
    seed_image =  fmnist_loader.get_x_test()[int(seed)]
    xml_desc = vectorization_tools.vectorize(seed_image, is_fmnist=True)
    return Digit(xml_desc, EXPECTED_LABEL, seed)

# get predicitons and metrics for digits for input validation
def generate_and_evaluate_digit(seed):
    seed_image = fmnist_loader.get_x_test()[int(seed)]
    xml_desc = vectorization_tools.vectorize(seed_image, is_fmnist=True)

    digit =  Digit(xml_desc, EXPECTED_LABEL, seed)

    predicted_label, confidence = predictor.Predictor.predict(digit.purified)

    digit.confidence = confidence
    digit.predicted_label = predicted_label

    return digit


def get_seeds_class(exp_label):
    # get seeds for expected class
    _, labels = fmnist_loader.get_x_test(), fmnist_loader.get_y_test()

    # Filter images belonging to class 1 (Trousers) and class 2 (Pullover)
    ind_seeds = np.where(labels == exp_label)[0] 

    return ind_seeds
