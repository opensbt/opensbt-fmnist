from itertools import permutations
from typing import List, Tuple
import logging as log
from pymoo.core.individual import Individual

import numpy as np
from typing import List

from enum import Enum

class ProcessingMode(Enum):
    BULK = 1
    SEQUENTIAL = 2

def euclidean_dist(ind_1, ind_2):
    return np.linalg.norm(ind_1.get("X") - ind_2.get("X"))

class IndividualSet(set):
    pass

class Archive(IndividualSet):
    def process_population(self, pop: List):
        raise NotImplemented()
    
class SmartArchiveInput(List):
    def __init__(self, archive_threshold):
        super().__init__()
        self.archive_threshold = archive_threshold

    def closest_individual_from_ind(self, ind):
        if len(self) == 0:
            return None, 0
        else:
            closest_ind = None
            closest_dist = 10000
            for ind_other in self:
                dist = euclidean_dist(ind, ind_other)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_ind = ind_other
            return (closest_ind, closest_dist)
        
    def closest_individual_from_vars(self, variables):
        ind = Individual()
        ind.set("X", variables)
        return self.closest_individual_from_ind(ind)
    
    def process_population(self, pop, mode: ProcessingMode):
        
        if mode == ProcessingMode.BULK:
            # check for all at the same time if they can be added, add at the end
            ind_to_add = []
            for ind in pop:
                if len(self) == 0:
                    ind_to_add.append(ind)
                else:
                    closest_archived, candidate_archived_distance = self.closest_individual_from_ind(ind)
                    # print(f"candidate_archived_distance is: {candidate_archived_distance}")
                    if candidate_archived_distance > self.archive_threshold:
                        ind_to_add.append(ind)
            
            for ind in ind_to_add:
                self._int_add(ind)

        elif mode == ProcessingMode.SEQUENTIAL:
            for ind in pop:
                self.process_individual(ind)
        else:
            print(f"Mode {mode} not known. Available modes: {ProcessingMode.BULK}, {ProcessingMode.SEQUENTIAL}")
            raise Exception()
        
    def process_individual(self, candidate):
        if len(self) == 0:
            self._int_add(candidate)
            log.debug('add initial individual')
        else:
            # uses semantic_distance to exploit behavioral information
            closest_archived, candidate_archived_distance = self.closest_individual_from_ind(candidate)
            # print(f"candidate_archived_distance is: {candidate_archived_distance}")
            if candidate_archived_distance > self.archive_threshold:
                log.debug('candidate is far from any archived individual')
                self._int_add(candidate)
                print('added to archive')
            else:
                print('closest archived is too close, dont add')

    def _int_add(self, candidate):
        self.append(candidate)
        # print('archive add', candidate)