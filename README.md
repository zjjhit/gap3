# Generic Algorithm for Predictive Probability guided Prompting (GAP3)

Source code for the IJCAI 2023 paper: "Genetic Prompt Search via Exploiting Language Model Probabilities"

#### Model
Discontinuous prompt chunks are considered as chromosomes, with prompt tokens being genes. Then, starting from an empty one, GAP3 evolves the prompts via chromosome crossovers and gene mutations. At each mutation step, either a new mask token is inserted into a random chromosome at a random position, or a random existing gene is masked. After this, the masked slot will be filled by a token that approximately maximises the predictive probability of the ground-truth labels on a (few-shot) training set. The algorithm iterates for a predefined number of steps, with individuals consisting of diverse
chromosomes/genes competing to survive and breed, according to their fitness scores computed on the training set.

#### Requirements
`pip install -r requirements.txt`

#### Quick Start
1. For RoBERTa-based experiments, run the scripts via `bash run_roberta.sh [random_seed] [task_name]`.
2. For GPT2-based experiments, run the scripts via `bash run_gpt.sh [random_seed] [task_name]`.    
3. Important arguments:
   * `--task_name`: The name of a task. choices = `[mrpc, snli, sst2, rte, yelpp, agnews, dbpedia]`.
   * `--k_shot`: number of shots.
   * `--petri_size`: number of individuals in each generation.
   * `--petri_iter_num`: number of generations/iterations.
   * `--mutate_prob`: token mutation probability.
   * `--crossover_prob`: chromosome crossover probability.

#### Contact information

For help or issues using GAP3, please submit a GitHub issue.

For personal communication related to GAP3, please contact Jiangjiang Zhao (`zhaojiangjiang@cmos.chinamobile.com`) or Zhuoran Wang (`wangzhuoran@clouchie.ai`).

#### Citation

If you use or extend our work, please cite the following paper:
```
@inproceedings{zhao2023gap3,
  title={Genetic Prompt Search via Exploiting Language Model Probabilities},
  author={Zhao, Jiangjiang and Wang, Zhuoran and Yang, Fangchun},
  booktitle={Proceedings of IJCAI 2023},
  year={2023}
}
```
   