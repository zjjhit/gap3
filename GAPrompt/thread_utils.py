import os
from queue import Queue
from threading import Thread, Lock
import torch
import copy
import traceback
from .data_utils import add_prompt_ids_to_data
from .utils import Cache
from .data_utils import LABEL_POSITION

cache_lock = Lock()
global_cache = Cache()

global_bias = None
bias_lock = Lock()

class SampleTokenWorker(Thread):
    def __init__(self, queue, device_id, model, data_sample, args):
        Thread.__init__(self)
        self.queue = queue
        self.device_id = device_id
        self.model = copy.copy(model)
        self.model.set_device(device_id)
        self.data_sample = data_sample
        self.args = args

        global global_bias
        if global_bias == None:
            global_bias = torch.zeros(self.model.tokenizer.vocab_size)
    
    def run(self):

        global global_cache
        global cache_lock

        global global_bias

        while True:
            try:
                individual = self.queue.get(block=False)
            except:
                break
            
            try:
                masked_data, _ = add_prompt_ids_to_data(
                    data = self.data_sample,
                    prompts_ids = individual.chromosomes,
                    with_labels = True,
                    cls_token_id = self.model.tokenizer.cls_token_id,
                    sep_token_id = self.model.tokenizer.sep_token_id,
                    pad_token_id = self.model.tokenizer.pad_token_id,
                    max_length=self.args.max_seq_length,
                )
                unlabeled_data, labels = add_prompt_ids_to_data(
                    data = self.data_sample,
                    prompts_ids = individual.chromosomes,
                    with_labels = 'mask',
                    cls_token_id = self.model.tokenizer.cls_token_id,
                    sep_token_id = self.model.tokenizer.sep_token_id,
                    pad_token_id = self.model.tokenizer.pad_token_id,
                    mask_token_id= self.model.place_holder_id,
                    max_length=self.args.max_seq_length,
                )

                if self.args.sample_tokens:
                    (
                        tokens, 
                        label_log_probs, 
                        token_log_probs, 
                        unlabel_token_log_probs
                    ) = self.model.sample_tokens(self.args.petri_size**2,masked_data,unlabeled_data,set(labels),global_bias)
                else:
                    (
                        tokens, 
                        label_log_probs, 
                        token_log_probs, 
                        unlabel_token_log_probs
                    ) = self.model.get_max_prob_tokens(self.args.petri_size**2,masked_data,unlabeled_data,set(labels),global_bias)

                sum_label_log_probs = torch.sum(torch.index_select(label_log_probs.cpu(),-1,torch.tensor(labels))).item()
                chromosomes = individual.chromosomes
                c, t = None, None
                for i in range(len(chromosomes)):
                    if chromosomes[i] is None:
                        continue
                    for j in range(len(chromosomes[i])):
                        if chromosomes[i][j] == self.model.tokenizer.mask_token_id:
                            c = i
                            t = j
                            break
                    if c != None and t != None:
                        break
                if c == None or t == None:
                    print("No mask token found",flush=True)
                    continue
                current = 0
                while current < len(tokens):
                    individual.chromosomes[c][t] = tokens[current]
                    if global_cache.is_cached(individual):
                        current += 1
                        continue
                    individual.new_token_id = tokens[current]
                    individual.left_log_prob = sum_label_log_probs+token_log_probs[individual.new_token_id].item()
                    individual.right_log_token_prob = unlabel_token_log_probs[individual.new_token_id].item()
                    cache_lock.acquire()
                    global_cache.add(individual)
                    cache_lock.release()
                    break

            except: 
                traceback.print_exc()

            finally:

                self.queue.task_done()


class FitWorker(Thread):
    def __init__(self, queue, device_id, model, train_data):
        Thread.__init__(self)
        self.queue = queue
        self.device_id = device_id
        self.model = copy.copy(model)
        self.model.set_device(device_id)
        self.train_data = train_data
        #self.dev_data = dev_data
    
    def run(self):
        
        global bias_lock
        global global_bias

        while True:
            try:
                individual = self.queue.get(block=False)
            except:
                break
            
            try:
                label_log_probs = self.model.fitness([individual],self.train_data)
                if individual.new_token_id != None:
                    sum_label_log_probs = torch.sum(label_log_probs).item()
                    bias_lock.acquire()
                    delta = sum_label_log_probs + individual.right_log_token_prob - individual.left_log_prob
                    global_bias[individual.new_token_id] = min(global_bias[individual.new_token_id],delta)
                    bias_lock.release()
                individual.new_token_id = None
            except: 
                traceback.print_exc()

            finally:
                self.queue.task_done()