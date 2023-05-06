
import numpy as np
from GAPrompt.arguments import get_args
from GAPrompt.modelwrapper import SampleModelWrapper, FitnessModelWrapper, label_to_token_id
from GAPrompt.petri import PetriDish, init_population
from GAPrompt.data_utils import load_raw_data, SET_LABEL_POSITION
from GAPrompt.utils import set_seed, list_skip_none

from task_config import *

import warnings
warnings.filterwarnings('ignore') 

class Verifier:
    def __init__(self,tokenizer,max_length,keys) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.keys = keys

    def verify(self,example):
        full_text = ''
        for i,k in enumerate(self.keys):
            if i == 0:
                full_text += str(example[k]) 
            else:
                full_text += ' '+str(example[k])
        features = self.tokenizer(full_text)
        return len(features['input_ids']) < self.max_length 

class Cutter:
    def __init__(self,tokenizer,max_length,keys) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.keys = keys

    def cut(self,example):
        full_text = ''
        for i,k in enumerate(self.keys):
            if i == 0:
                full_text += str(example[k]) 
            else:
                full_text += ' '+str(example[k])
        features = self.tokenizer(full_text)
        full_len = len(features['input_ids'])
        if full_len >= self.max_length:
            longest = 0
            key_longest = None
            for k in self.keys:
                if k == 'label':
                    continue
                if len(example[k]) > longest:
                    longest = len(example[k])
                    key_longest = k
            
            text_to_cut = example[key_longest]
            features = self.tokenizer(text_to_cut,add_special_tokens=False)
            new_text = self.tokenizer.decode(features['input_ids'][:self.max_length-full_len-1])
            example[key_longest] = new_text
        return example

def cut_long_sentences(raw_dataset,cutter):
    new_dataset = []
    for example in raw_dataset:
        new_dataset.append(cutter.cut(example))
    return new_dataset

def load_data_as_text_chunks(data,keys,labels):
    ans = []
    for i in range(len(data)):
        example = []
        for x in keys:
            s = data[i][x]
            if labels and isinstance(s,int):
                s = labels[s]
            example.append(s)
        ans.append(example)
    return ans

def construct_true_few_shot_data(task_name, split, k_shot, verifiers=None):
    
    train_label_count = {}
    dev_label_count = {}
    new_train_data = []
    new_dev_data = []

    train_data = load_raw_data(task_to_names[task_name],split,task_to_datafiles[task_name])

    all_indices = [_ for _ in range(len(train_data))]
    np.random.shuffle(all_indices)

    for index in all_indices:
        label = train_data[index]['label']
        if label < 0:
            continue

        if label not in train_label_count:
            train_label_count[label] = 0
        if label not in dev_label_count:
            dev_label_count[label] = 0

        if train_label_count[label] < k_shot:
            #if verifier and not verifier.verify(train_data[index]):
            if verifiers:
                skip = False
                for verif in verifiers:
                    if not verif.verify(train_data[index]):
                        skip = True
                if skip:
                    continue
            new_train_data.append(train_data[index])
            train_label_count[label] += 1
        elif dev_label_count[label] < k_shot:
            if verifiers:
                skip = False
                for verif in verifiers:
                    if not verif.verify(train_data[index]):
                        skip = True
                if skip:
                    continue
            new_dev_data.append(train_data[index])
            dev_label_count[label] += 1
    
    train_text_chunks = load_data_as_text_chunks(new_train_data,task_to_keys[task_name],task_to_labels[task_name])
    dev_text_chunks = load_data_as_text_chunks(new_dev_data,task_to_keys[task_name],task_to_labels[task_name])

    return train_text_chunks, dev_text_chunks

def check_hyper_params(template,args):
    length = 0
    for e in template:
        if e is not None:
            length += 1
    if length == 1:
        args.crossover_prob = 0
        args.mutate_prob = 1

def move_labels_to_the_end(label_list):
    label_list.remove('label')
    label_list.append('label')
    return label_list

args = get_args()

set_seed(args.seed)

if 'gpt' in args.eval_model:
    task_to_keys[args.task_name] = move_labels_to_the_end(task_to_keys[args.task_name])

SET_LABEL_POSITION(task_to_keys[args.task_name].index('label'))

check_hyper_params(task_to_prompt_templates[args.task_name],args)

#batch_size = min(args.k_shot * len(task_to_labels[args.task_name]), 64)


batch_size = args.batch_size

sample_model = SampleModelWrapper(args.mlm_model,batch_size=batch_size)
fitness_model = FitnessModelWrapper(args.eval_model,max_length=args.max_seq_length,batch_size=batch_size)

if task_to_metric[args.task_name] != None:
    metric_class, pos_label_index = task_to_metric[args.task_name]
    pos_label = label_to_token_id(task_to_labels[args.task_name][pos_label_index],fitness_model.tokenizer)
    fitness_model.set_custom_metric( metric_class(pos_label) )

verif1 = Verifier(
    tokenizer = sample_model.tokenizer,
    max_length = 512 - args.petri_iter_num*2,
    keys = task_to_keys[args.task_name],
)

verif2 = Verifier(
    tokenizer = fitness_model.tokenizer,
    max_length = 512 - args.petri_iter_num*2,
    keys = task_to_keys[args.task_name],
)

verifiers = [verif1]
if args.mlm_model != args.eval_model:
    verifiers.append(verif2)

train_data, dev_data = construct_true_few_shot_data(
    args.task_name,
    task_to_dataset_tags[args.task_name][0],
    args.k_shot,
    verifiers=verifiers,
)


print('# of train data: {}'.format(len(train_data)), flush=True)
print('Example:', flush=True)
print(train_data[0], flush=True)

print('# of dev data: {}'.format(len(dev_data)), flush=True)
print('Example:', flush=True)
print(dev_data[0], flush=True)


population = init_population(
    task_to_prompt_templates[args.task_name],
    args.petri_size,
    sample_model.tokenizer.mask_token_id
)

petri = PetriDish(train_data,population,sample_model,fitness_model,args)

petri.evolute(args.crossover_prob,args.mutate_prob,args.petri_iter_num)


best_on_training_only = petri.best_individual


raw_test_data = load_raw_data(
    task_to_names[args.task_name],
    task_to_dataset_tags[args.task_name][1],
    task_to_datafiles[args.task_name],
)


prompt_len = len(
    fitness_model.tokenizer(
        [
            ' '.join(list_skip_none(best_on_training_only.prompts))
        ],
        add_special_tokens=False,
        padding=True
    )['input_ids'][0]
)

cutter = Cutter(
    tokenizer = fitness_model.tokenizer,
    max_length = 512 - prompt_len,
    keys = task_to_keys[args.task_name],
)

raw_test_data = cut_long_sentences(raw_test_data,cutter)

test_data = load_data_as_text_chunks(
    raw_test_data,
    task_to_keys[args.task_name],
    task_to_labels[args.task_name],
)

print('\n# of test data: {}'.format(len(test_data)), flush=True)
print('Example:', flush=True)
print(test_data[0], flush=True)



petri.test([best_on_training_only],test_data)
print(">>>Test accuracy: {}".format(best_on_training_only.accuracy),flush=True)
print('>>>Prompt: '+' ||| '.join(list_skip_none(best_on_training_only.prompts)),flush=True)


