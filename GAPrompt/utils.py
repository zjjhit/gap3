
import math
import random
import numpy as np
import torch

class Cache:
    def __init__(self):
        self.cache = dict()
    def add(self,individual):
        key = individual.as_key()
        if key not in self.cache:
            self.cache[key] = individual
    def is_cached(self,individual):
        return individual.as_key() in self.cache        

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def map_to_prob_intervals(population, use_exp = False):
    total = 0
    for ind in population:
        score = ind.accuracy
        if use_exp:
            score = math.exp(score)
        total += score
    
    intervals = []
    for ind in population:
        score = ind.accuracy
        if use_exp:
            score = math.exp(score)
        intervals.append(score/total)
    return intervals

def sample(population, use_exp = False):
    intervals = map_to_prob_intervals(population,use_exp)
    p = random.uniform(0,1)
    accum = 0
    for i,v in enumerate(intervals):
        accum += v
        if accum > p:
            return i
    return None

def sample_pair(population, use_exp = False):
    first = sample(population,use_exp)
    second = sample(population,use_exp)
    max_steps = 10
    try_steps = 0
    while second == first and try_steps < max_steps:
        second = sample(population,use_exp)
        try_steps += 1

    return population[first], population[second]

def list_skip_none(list_like):
    ans = []
    for e in list_like:
        if not e is None:
            ans.append(e)
    return ans

def random_choice_skip_none(list_like):
    return random.choice(list_skip_none(list_like))
    '''
    length = 0
    for e in list_like:
        if e is None:
            continue
        length += 1
    
    i = random.randint(0,length-1)
    j = 0
    for e in list_like:
        if e is None:
            continue
        if j == i:
            return e
        j += 1
    return None
    '''

#def sort_by(individuals):
#    individuals.sort(key=lambda x: x.fitness_score, reverse=True)

'''
def two_stage_tournament(population,fitness_model,n_groups,n_samples,n_top,n_elites):
    pop_size = len(population)
    group_size = population//n_groups
    if population % n_groups != 0:
        n_groups += 1
    
    #first stage: group competition
    random.shuffle(population)

    to_the_final = []

    for i in range(n_groups):
        group = population[i*group_size:min((i+1)*group_size,pop_size)]
        if len(group) > n_top:
            fitness_model.stage_fitness(group,n_samples)
            sort_by(group,'stage')
            to_the_final += group[:n_top]
        else:
            to_the_final += group
    
    #second stage: the final tournament
    fitness_model.fitness(to_the_final)
    sort_by(to_the_final,'fitness')
    return to_the_final[:n_elites]
'''