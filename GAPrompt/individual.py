# coding=utf-8

import random
import copy
from .utils import random_choice_skip_none

def insert_left(chromosome,i,mask_id):
    chromosome.insert(i,mask_id)

#def insert_right(chromosome,i,mask_id):
#    chromosome.insert(i+1,mask_id)

#def delete(chromosome,i,mask_id):
#    del chromosome[i]

def replace(chromosome,i,mask_id):
    chromosome[i] = mask_id

OPERATIONS = [insert_left,replace]
OPERATION_PROBS = [0.5,0.5]

def sample_operation():
    p = random.uniform(0,1)
    accum = 0
    for i , prob in enumerate(OPERATION_PROBS):
        accum += prob
        if accum > p:
            return i #OPERATIONS[i]
    return len(OPERATIONS)-1 #OPERATIONS[len(OPERATIONS)-1]

class Individual:
    '''
    Class representing individual in population
    '''
    def __init__(self, chromosomes, abstract=False):
        self.chromosomes = chromosomes
        self.fitness_score = None
        self.abstract = abstract
        self.prompts = []
        self.accuracy = None
        self.new_token_id = None
        self.left_log_prob = 0
        self.right_log_token_prob = 0

    def mutate(self, mutate_prob, mask_token_id):
        '''
        create random genes for mutation
        ''' 
        p = random.uniform(0,1)
        if p >= mutate_prob:
            return

        chr = random_choice_skip_none(self.chromosomes)

        self.abstract = True
        if len(chr)==0:
            chr.append(mask_token_id)
            return
        
        op = sample_operation()
        #op(chr,j,mask_token_id)        
        if op == 0:
            j = random.randint(0,len(chr))
        else:
            j = random.randint(0,len(chr)-1)
        
        func = OPERATIONS[op]
        func(chr,j,mask_token_id)

    def crossover(self, partner, crossover_prob):
        '''
        Perform mating and produce new offspring
        '''
        new_self = Individual(copy.deepcopy(self.chromosomes))
        new_partner = Individual(copy.deepcopy(partner.chromosomes))
        for i,chr in enumerate(new_self.chromosomes):
            if chr is None:
                continue
            p = random.uniform(0,1)
            if p < crossover_prob:
                tmp = chr
                new_self.chromosomes[i] = new_partner.chromosomes[i]
                new_partner.chromosomes[i] = tmp
        
        return new_self,new_partner

    def as_key(self):
        #print(self.chromosomes)
        key = []
        for x in self.chromosomes:
            if x is None:
                key.append('')
            else:
                key.append(' '.join(str(e) for e in x))
       
        return '\x01'.join(key)
