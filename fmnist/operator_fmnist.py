import numpy as np
from fmnist.fmnist_problem import FMNISTProblem
from opensbt.evaluation.fitness import *
import sys
import numpy as np
from pymoo.core.sampling import Sampling
from mnist import mutations
from pymoo.operators.sampling.rnd import random_by_bounds

class FMnistSamplingValid(Sampling):

    def _do(self, problem, n_samples, **kwargs):

        X = np.empty((n_samples, len(problem.xu)))
        
        seed_pure= problem.seed_digits[0]
    
        for i in range(0,n_samples):
            while True:
                cand = random_by_bounds(n_var=problem.n_var,
                                n_samples=1,
                                xl=problem.xl,
                                xu=problem.xu)[0]
                # print(f"created candidate: {cand}")
                
                if problem.n_var == 3:
                    digit_mutated = self._generate_digit_mutated(problem,
                                    digit=seed_pure,
                                    extent_1=cand[0],
                                    extent_2=cand[1],
                                    c_index=cand[2])
                elif problem.n_var == 6:
                    digit_mutated = self._generate_digit_mutated_bi(problem, 
                            seed_pure, 
                            cand[0], 
                            cand[1], 
                            cand[2],
                            cand[3],
                            cand[4],
                            cand[5])
                else:
                    print("Problem instance not supported. N vars is not supported.")
                    sys.exit()

                if digit_mutated.distance(seed_pure) <= problem.max_seed_distance:
                    break
                else:
                    print("Digit invalid. Resampling...")
            X[i] = np.asarray(cand)
        # print(f"constructed initial population: {X}")

        return X

    def _generate_digit_mutated(self, 
            problem: FMNISTProblem, 
            digit, 
            extent_1, 
            extent_2, 
            c_index):
        print("Generating new digit by mutation.")
        new_digit = digit.clone()
        new_digit = mutations.apply_mutation_index(problem,
                                                          new_digit,
                                                            extent_1=extent_1,
                                                            extent_2=extent_2,
                                                            index=c_index)

        return new_digit

    def _generate_digit_mutated_bi(self, 
            problem: FMNISTProblem, 
            digit, 
            extent_1, 
            extent_2, 
            extent_3,
            extent_4,
            c_index_1,
            c_index_2):
        print("Generating new digit by mutation.")
        new_digit = digit.clone()
        new_digit = mutations.apply_mutation_index_bi(
                            problem,
                            new_digit,
                            extent_1=extent_1,
                            extent_2=extent_2,
                            extent_3=extent_3,
                            extent_4=extent_4,
                            index_1=c_index_1,
                            index_2=c_index_2
                        )

        return new_digit