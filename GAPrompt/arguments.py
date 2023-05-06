# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""argparser configuration"""

import argparse
import os
import torch

def add_model_config_args(parser):
    """Model arguments"""

    group = parser.add_argument_group('model', 'model configuration')
    group.add_argument('--mlm_model', type=str, default='roberta-large',
                       help="mask LM name")
    group.add_argument('--eval_model', type=str, default='roberta-large',
                       help="Fitness PLM name")
                       
    group.add_argument('--init_pop_size', type=int, default=1000,
                       help="initial population size")

    group.add_argument('--petri_size', type=int, default=64,
                       help="number of individuals in each petri")

    group.add_argument('--petri_iter_num', type=int, default=50,
                       help="number of evoluting iterations for each petri")

    group.add_argument('--batch_size', type=int, default=32, help="batch_size")
    
    group.add_argument('--max_seq_length', type=int, default=512,
                       help="maximum sequence length for LMs")

    group.add_argument('--mutate_prob', type=float, default=0.75,
                       help="gene mutation probability")

    group.add_argument('--crossover_prob', type=float, default=0.5,
                       help="chromosome crossover probability")

    group.add_argument('--seed', type=int, default=42,
                       help="random seed")

    group.add_argument('--sample_tokens', action='store_true',
                       help="whether to sample tokens when realizing each petri or to get those with maximum probabilities")
    
    group.add_argument('--log_dir', type=str, default='log',
                       help="directory to store thread log files")

    group.add_argument('--warmup', type=int, default=10,
                       help="warmup evolutions")
    
    group.add_argument('--top_k_dev', type=int, default=1,
                       help="top k individual to verify on dev set")
    
    return parser

def add_data_args(parser):
    """Train/valid/test data arguments."""

    group = parser.add_argument_group('data', 'data configurations')
    group.add_argument('--task_name', type=str, default=None,
                       help="the task name")
    
    group.add_argument('--data_size_for_debug', type=int, default=None,
                       help="number of training examples for debugging")
    
    group.add_argument('--k_shot', type=int, default=32,
                       help="number of training examples for k-shot learning")

    group.add_argument('--do_dev', action='store_true',
                        help="do validation after training")

    group.add_argument('--do_test', action='store_true',
                        help="run test set after training")

    group.add_argument('--output', type=str, default='test.out',
                        help="output file to write test set predictions")

    group.add_argument('--test_prompt', type=str, default=None,
                       help="pretained prompt (in text file)")

    return parser


def get_args():
    """Parse all the args."""

    parser = argparse.ArgumentParser(description='GAPO: GA Prompt Optimisation')
    parser = add_model_config_args(parser)
    parser = add_data_args(parser)

    args = parser.parse_args()

    return args
