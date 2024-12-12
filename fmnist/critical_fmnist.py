import numpy as np
from opensbt.simulation.simulator import SimulationOutput
from opensbt.evaluation.critical import Critical

class CriticalFMNIST(Critical):
    ''' Conf_diff_max < 0, Conf_missclassified_label > 0.5'''
    def eval(self, vec_fit: np.ndarray, simout: SimulationOutput) -> bool:
        # return vec_fit[0] < 0 and vec_fit[1] < 0.1  # Conf diff, Conf miss
        # return vec_fit[0] < 0 and (vec_fit[1] == -7) # Conf diff, Predicted
        # return vec_fit[0] < -0.2 and vec_fit[1] < -1 # Conf diff, Dist archive # Random value, need some method/explanation to choose it.
        # return vec_fit[0] < 0 and (vec_fit[1] < -0.5) # Conf diff, maximize prediction as 8
        return vec_fit[0] < -0.9 and vec_fit[1] < 0.2 # Conf diff, Dist archive # Random value, need some method/explanation to choose it.
        # return vec_fit[0] < 0.5 and vec_fit[1] < 0.6 # Conf diff, Dist archive # Random value, need some method/explanation to choose it.
        #return vec_fit[0] < 0.2 and vec_fit[1] > 0.5 and vec_fit[1] < 0.55  # Conf diff, Dist archive # Random value, need some method/explanation to choose it.
