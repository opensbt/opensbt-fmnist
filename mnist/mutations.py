import sys
from model_ga.population import PopulationExtended
from mnist.digit_input import Digit
from mnist.digit_mutator import DigitMutator
from mnist.config import EXPECTED_LABEL
from scipy.stats import entropy
from pathlib import Path
import numpy as np
import logging as log

def apply_mutation_index_bi(problem, new_digit, extent_1, extent_2,extent_3, extent_4, index_1, index_2):  
        # assure that x,y coordinage of the same point are mutated
        if index_1 % 2 == 0:
            next_vertex = index_1 + 1
        else:
            next_vertex = index_1- 1
        # next_vertex = index + 1
        DigitMutator(new_digit).mutate(extent_1, c_index = index_1, mutation=2)
        DigitMutator(new_digit).mutate(extent_2, c_index = (next_vertex % problem.vertex_num), mutation=2)
        
        # DigitMutator(new_digit).mutate(extent_1, c_index = index + 2, mutation=2)
        # DigitMutator(new_digit).mutate(extent_2, c_index = (next_vertex + 2 % problem.vertex_num), mutation=2)
        if index_2 % 2 == 0:
            next_vertex = index_2 + 1
        else:
            next_vertex = index_2 - 1
        # Perform displacement on inner point
        DigitMutator(new_digit).mutate(extent_3, c_index = index_2, mutation=1)
        DigitMutator(new_digit).mutate(extent_4, c_index = (next_vertex % problem.vertex_num), mutation=1)
        
        # Perform displacement on inner point
        # DigitMutator(new_digit).mutate(extent_1, c_index = (index + 2 ) % problem.vertex_num, mutation=1)
        # DigitMutator(new_digit).mutate(extent_2, c_index = ((next_vertex + 2) % problem.vertex_num), mutation=1)

        return new_digit

def apply_mutation_index(problem, new_digit, extent_1, extent_2, index):  
    # assure that x,y coordinage of the same point are mutated
    if index % 2 == 0:
        next_vertex = index + 1
    else:
        next_vertex = index - 1
    # next_vertex = index + 1

    DigitMutator(new_digit).mutate(extent_1, c_index = index, mutation=2)
    DigitMutator(new_digit).mutate(extent_2, c_index = (next_vertex % problem.vertex_num), mutation=2)
    
    # DigitMutator(new_digit).mutate(extent_1, c_index = index + 2, mutation=2)
    # DigitMutator(new_digit).mutate(extent_2, c_index = (next_vertex + 2 % problem.vertex_num), mutation=2)
    
    # Perform displacement on inner point
    DigitMutator(new_digit).mutate(extent_1, c_index = index, mutation=1)
    DigitMutator(new_digit).mutate(extent_2, c_index = (next_vertex % problem.vertex_num), mutation=1)
    
    # Perform displacement on inner point
    DigitMutator(new_digit).mutate(extent_1, c_index = (index + 2 ) % problem.vertex_num, mutation=1)
    DigitMutator(new_digit).mutate(extent_2, c_index = ((next_vertex + 2) % problem.vertex_num), mutation=1)

    return new_digit


def apply_mutation_segment(problem, new_digit, extent_1, extent_2, vertex):              
    DigitMutator(new_digit).mutate_point(extent_x=extent_1, 
                                            extent_y=extent_2, 
                                            segment=vertex, 
                                            mutation=1)
                                            
    DigitMutator(new_digit).mutate_point(extent_x= 0.5* extent_1, 
                                        extent_y= 0.5 * extent_2, 
                                        segment=(vertex + 3) % problem.segment_num, 
                                        mutation=2)

    DigitMutator(new_digit).mutate_point(extent_x=0.5* extent_1, 
                            extent_y=0.5* extent_2, 
                            segment=(vertex + 5) % problem.segment_num, 
                            mutation=2)
    return new_digit

def get_class_from_function(f):
    return vars(sys.modules[f.__module__])[f.__qualname__.split('.')[0]] 

def get_min_distance_from_archive(digit: Digit, archive: PopulationExtended):
    distances = []
    for archived_digit in archive.get("DIG"):
        # print("Digit or some close digit is in archive.")
        if archived_digit.purified is not digit.purified:
            dist = np.linalg.norm(archived_digit.purified - digit.purified)
            # TODO fix, distance is somehow 0, even when digit not in archive
            # if dist == 0:
            #     print("Distance is 0, skip.")
            #     continue
            distances.append(dist)
    if len(distances) == 0:
        return 0
    else:
        min_dist = min(distances)
    return min_dist

def get_min_distance_from_archive_input(X_ind, archive: PopulationExtended):
    distances = []
    for X in archive.get("X"):
        # print("Digit or some close digit is in archive.")
        if X_ind is not X:
            dist = np.linalg.norm(X - X_ind)
            distances.append(dist)
    if len(distances) == 0:
        return 0
    else:
        min_dist = min(distances)
    return min_dist
