from .data_utils import *
from .utils import *
import random
import math
from tqdm import tqdm
from .individual import Individual
from .modelwrapper import SampleModelWrapper, FitnessModelWrapper
import copy
import logging
from .thread_utils import SampleTokenWorker, FitWorker
from threading import Thread
from queue import Queue
import torch
from .thread_utils import global_cache
import time
from functools import cmp_to_key

def init_population(template,n_individuals,mask_token_id):
    population = []
    print("Sample {} individuals as the initial population...".format(n_individuals),flush=True)
    #cache = Cache()
    #while len(population) < n_individuals:
    for _ in range(n_individuals):
        chromosomes = copy.deepcopy(template)
        seed_chr = random_choice_skip_none(chromosomes)
        #seed_chr = random.choice(chromosomes)
        #while seed_chr is None:
        #    seed_chr = random.choice(chromosomes)
        seed_chr.append(mask_token_id)
        ind = Individual(chromosomes,True)
        population.append(ind)
        #cache.add(ind)
    return population

def better_than(ind1,ind2):
    if math.fabs(ind1.accuracy - ind2.accuracy) < 1e-8:
        return ind1.fitness_score - ind2.fitness_score
    else:
        return ind1.accuracy - ind2.accuracy

class PetriDish:
    def __init__(self,
        raw_train_data,
        #raw_dev_data,
        individuals,
        mlm_model,
        eval_model,
        args,
    ):
        self.raw_train_data = raw_train_data
        if raw_train_data and mlm_model:
            self.data_sample = tokenize_data_chunks(raw_train_data,mlm_model.tokenizer)
        #self.raw_dev_data = raw_dev_data
        self.individuals = individuals
        self.mlm_model = mlm_model
        self.eval_model = eval_model
        #self.cache = Cache()
        self.best_individual = None
        #self.log_file_handler = None
        self.logger = None
        self.args = args
        self.to_verify_on_dev = []
    
    def set_log_file(self,filename):
        log_file_handler = logging.FileHandler(filename)
        #f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        #log_file_handler.setFormatter(f_format)
        self.logger = logging.getLogger(filename)
        self.logger.addHandler(log_file_handler)

    def close_log(self):
        handlers = self.logger.handlers[:]
        for handler in handlers:
            self.logger.removeHandler(handler)
            handler.flush()
            handler.close()

    def release(self):
        self.close_log()
        self.raw_train_data = None
        #self.raw_dev_data = None
        self.data_sample = None
        self.mlm_model = None
        self.eval_model = None
        #self.cache = None
        self.logger = None
    
    def log(self,string,flush=True):
        if self.logger == None:
            print(string,flush=flush)
        else:
            self.logger.critical(string)

    def print(self):
        for i,s in enumerate(self.raw_train_data):
            self.log("{}: {}".format(i,s),flush=True)
        for i,d in enumerate(self.data_sample):
            self.log("{}: {}".format(i,d),flush=True)
        #for i,s in enumerate(self.raw_dev_data):
        #    self.log("{}: {}".format(i,s),flush=True)
        self.__print_individuals()
    
    def __print_individuals(self):
        self.decode_prompts(self.individuals)
        for i,ind in enumerate(self.individuals):
            self.log("{} : {} > {} {}".format(i,' ||| '.join(list_skip_none(ind.prompts)),ind.fitness_score,ind.accuracy),flush=True)

    def realize(self):
        n_threads = torch.cuda.device_count()
        to_realize_queue = Queue()
        for ind in self.individuals:
            if not ind.abstract:
                #population.append(ind)
                #self.cache.add(ind)
                #global_cache.add(ind)
                continue
            to_realize_queue.put(ind)
        #print("{} to realize".format(to_realize_queue.qsize()),flush=True)
        for i in range(n_threads):
            worker = SampleTokenWorker(to_realize_queue,i,self.mlm_model,self.data_sample,self.args)
            worker.daemon = True
            worker.start()

        to_realize_queue.join()

        #torch.cuda.empty_cache()

    def __label_statistics(self):
        def stat_dataset(dataset, name):
            
            counts = {}
            for example in dataset:
                label = example[GET_LABEL_POSITION()]
                if label not in counts.keys():
                    counts[label] = 1
                else:
                    counts[label] += 1
            self.log("{} Set:".format(name),flush=True)
            for k, v in counts.items():
                self.log("{}:\t{}\t{}".format(k,v,v/len(dataset)),flush=True)
        stat_dataset(self.raw_train_data,'Train')
        #stat_dataset(self.raw_dev_data,'Dev')
        
    
    def evolute(self, crossover_prob, mutate_prob, n_generations, debug=False):
        n_population = len(self.individuals)
        n_elites = int(math.sqrt(n_population))
        self.log("Label distribution:",flush=True)
        self.__label_statistics()
        
        self.realize()
        self.eval()

        #early_stopping_count = 0

        for i in tqdm(range(n_generations),desc ="Evolutions"):
            start_time = time.time()
            #if debug:
            self.log("Iteration {}:".format(i),flush=True)
            self.__print_individuals()

            elites = self.top(n_elites)
            if self.best_individual == None or self.best_individual.fitness_score == None or better_than(elites[0],self.best_individual):
                self.best_individual = elites[0]
                #early_stopping_count = 0
            else:
                elites.append(self.best_individual)
                #early_stopping_count += 1
            
            #if i >= self.args.warmup and early_stopping_count >= 2:
            #    self.log("Early stopping reached!!!")
            #    break
            
            if i >= self.args.warmup:
                self.to_verify_on_dev += elites[:self.args.top_k_dev]
            
            new_generation = []
            max_steps = n_population*10
            iter = 0
            while len(new_generation) < n_population and iter < max_steps:
                individual1, individual2 = sample_pair(elites,True)
                new_individuals = individual1.crossover(individual2, crossover_prob)
                if debug:
                    self.log("Crossover: ",flush=True)
                    self.log(individual1.chromosomes,flush=True)
                    self.log(individual2.chromosomes,flush=True)
                    self.log("Obtained:",flush=True)
                    for x in new_individuals:
                        self.log(x.chromosomes)
                for x in new_individuals:
                    if debug:
                        self.log("Mutate: ",flush=True)
                        self.log(x.chromosomes,flush=True)
                    x.mutate(mutate_prob,self.mlm_model.tokenizer.mask_token_id)
                    if debug:
                        self.log("Obtained:",flush=True)
                        self.log(x.chromosomes,flush=True)
                    #if not self.cache.is_cached(x):
                    #    new_generation.append(x)
                    #    self.cache.add(x)
                    #elif debug:
                    #    self.log('cached',flush=True)
                    if not x.abstract:
                        if global_cache.is_cached(x):
                            continue
                        global_cache.add(x)
                    
                    new_generation.append(x)
                iter += 1
            
            #if len(new_generation) < n_population:
            #    self.individuals = self.individuals + new_generation
            #    self.individuals = self.top(n_population,keep_the_best=False)
            #else:
            self.individuals = new_generation

            self.realize()
            self.eval()

            end_time = time.time()
            self.log("--- %s seconds ---" % (end_time - start_time),flush=True)

        #self.log('>>>Best score: {} accuracy: {}'.format(self.best_individual.fitness_score,self.best_individual.accuracy),flush=True)
        #self.log('>>>Prompt: '+' ||| '.join(self.best_individual.prompts),flush=True)

        last_best = self.top(1)[0]
        if self.best_individual == None or self.best_individual.fitness_score == None or better_than(last_best,self.best_individual):
            self.best_individual = last_best

    def top(self,k):

        if k > len(self.individuals):
            k = len(self.individuals)

        self.individuals = sorted(self.individuals, key=cmp_to_key(better_than), reverse=True)
        
        return self.individuals[:k]

    def verify_on_dev(self,raw_dev_data):
        self.decode_prompts(self.to_verify_on_dev)
        individuals_to_eval = Queue()
        for ind in self.to_verify_on_dev:
            individuals_to_eval.put(ind)
        n_threads = torch.cuda.device_count()
        for i in range(n_threads):
            worker = FitWorker(individuals_to_eval,i,self.eval_model,raw_dev_data)
            worker.daemon = True
            worker.start()
        individuals_to_eval.join()

    def best_on_dev(self,raw_dev_data):
        self.verify_on_dev(self.raw_train_data + raw_dev_data)
        rank_list = sorted(self.to_verify_on_dev, key=cmp_to_key(better_than), reverse=True)
        return rank_list[0]

    def decode_prompts(self,individuals):
        if not self.mlm_model:
            return
        for ind in individuals:
            ind.prompts = []
            for chr in ind.chromosomes:
                if chr is None:
                    ind.prompts.append(None)
                else:
                    prompt = self.mlm_model.tokenizer.decode(chr)
                    #if prompt.startswith('##'):
                    #    prompt = prompt[2:]
                    ind.prompts.append(prompt)
        
    def eval(self):
        self.decode_prompts(self.individuals)
        individuals_to_eval = Queue()
        need_fit = False
        for ind in self.individuals:
            if ind.fitness_score == None:
                individuals_to_eval.put(ind)
                need_fit = True
        if need_fit: 
            n_threads = torch.cuda.device_count()
            for i in range(n_threads):
                worker = FitWorker(individuals_to_eval,i,self.eval_model,self.raw_train_data)
                worker.daemon = True
                worker.start()
            individuals_to_eval.join()

    def test(self,individuals,raw_test_data):
        individuals_to_eval = Queue()
        for ind in individuals:
            individuals_to_eval.put(ind)
        n_threads = torch.cuda.device_count()
        for i in range(n_threads):
            worker = FitWorker(individuals_to_eval,i,self.eval_model,raw_test_data)
            worker.daemon = True
            worker.start()
        individuals_to_eval.join()

class PetriDishGenerator:
    def __init__(self,
        raw_train_set,
        keys,
        labels,
        mlm_model,
        eval_model,
        args,
    ):
        self.raw_dataset = load_data_as_text_chunks(
            raw_train_set,
            keys,
            labels,
            data_size_for_debug=args.data_size_for_debug
        )
        self.mlm_model = mlm_model
        self.eval_model = eval_model
        self.args = args

    def generate_petri_dishes(self,population):
        n_petri_dishes = len(population)//self.args.petri_size
        n_petri_size = len(self.raw_dataset)//n_petri_dishes
        random.shuffle(self.raw_dataset)
        random.shuffle(population)
        petri_dishes = []
        for i in tqdm(range(n_petri_dishes),desc="Generating petri dishes"):
            sub_pop = population[i*self.args.petri_size:(i+1)*self.args.petri_size]
            sub_data = self.raw_dataset[i*n_petri_size:(i+1)*n_petri_size]
            dish = PetriDish(sub_data,sub_pop,self.mlm_model,self.eval_model,args=self.args)
            petri_dishes.append(dish)
        return petri_dishes
