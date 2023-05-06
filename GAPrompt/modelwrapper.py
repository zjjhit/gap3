import torch
from transformers import AutoModelWithLMHead, AutoTokenizer
from transformers import BertTokenizer, BertForMaskedLM, RobertaForMaskedLM, T5ForConditionalGeneration, GPT2LMHeadModel
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM
from torch.nn import functional as F
import copy
import numpy as np
from .data_utils import add_prompts_to_data

def convert_to_t5_inputs(dataset,tokenizer,max_length):
    inputs = tokenizer(dataset,max_length=max_length,pad_to_max_length=True,truncation=True)
    decode_inputs = tokenizer(['<pad> <extra_id_0> <extra_id_1>']*len(dataset))
    inputs['decoder_input_ids'] = decode_inputs['input_ids'],
    inputs['decoder_attention_mask'] = decode_inputs['attention_mask']
    return inputs

def label_to_token_id(label,tokenizer):
    return tokenizer.encode(' '+label.strip(),add_special_tokens=False)[0]

class SampleModelWrapper:

    ModelPool = {}

    def __init__(self,model_name_or_instance,device=None,batch_size=128):
        if isinstance(model_name_or_instance,str):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_instance,use_fast=False)
            if 't5' in  model_name_or_instance:
                self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_instance)
            else:
                self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_instance)
                if 'roberta' in model_name_or_instance:
                    self.model.lm_head.bias = torch.nn.parameter.Parameter(torch.zeros(self.tokenizer.vocab_size))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_instance.tokenizer_name)
            self.model = model_name_or_instance

        if 'mask_token' not in self.tokenizer.special_tokens_map:
            #for T5
            self.tokenizer.add_special_tokens({'mask_token':'<extra_id_0>'})
    
        self.mask_token_id = self.tokenizer.mask_token_id
        self.place_holder_id = -1
        self.device = device if device else torch.device('cpu')
        self.batch_size = batch_size
        self.model.to(self.device)
        self.model.eval()
        self.offload_model = self.model
        self.offload_device = self.device

        self.label_mask_token_id = self.mask_token_id
        if self.__is_t5(): 
            self.label_mask_token_id = self.tokenizer.additional_special_tokens_ids[1]
            self.decode_inputs = self.tokenizer(['<pad> <extra_id_0> <extra_id_1>'])
            self.place_holder_id = self.label_mask_token_id
            

    def __is_t5(self):
        return isinstance(self.model,T5ForConditionalGeneration)

    def __convert_to_t5_inputs(self,inputs):
        #decode_inputs = self.tokenizer(['<pad> <extra_id_0> <extra_id_1>']*len(inputs['input_ids']))
        bsz = inputs['input_ids'].size(0)
        inputs['decoder_input_ids'] = torch.tensor(
                self.decode_inputs['input_ids']
            ).repeat(bsz,1).to(self.device)
        inputs['decoder_attention_mask'] = torch.tensor(
                self.decode_inputs['attention_mask']
            ).repeat(bsz,1).to(self.device)
        return inputs

    def set_device(self,device_id):
        device = torch.device("cuda:{}".format(device_id)) 
        if device_id not in SampleModelWrapper.ModelPool.keys():
            model = copy.deepcopy(self.offload_model)
            model.to(device)
            model.eval()
            SampleModelWrapper.ModelPool[device_id] = model
        self.model = SampleModelWrapper.ModelPool[device_id]
        self.device = device
    
    def offload(self):
        self.model = self.offload_model
        self.device = self.offload_device

    def __eval_batch(self,input):
        
        if self.__is_t5():
            mask_index = torch.where(input["decoder_input_ids"] == self.mask_token_id)
            label_mask_index = torch.where(input["decoder_input_ids"] == self.label_mask_token_id)
        else:
            mask_index = torch.where(input["input_ids"] == self.mask_token_id)
            label_mask_index = torch.where(input["input_ids"] == self.place_holder_id)
            input["input_ids"][label_mask_index] = self.label_mask_token_id

        with torch.no_grad():
            output = self.model(**input)
            logits = output.logits
            log_softmax_probs = torch.log(F.softmax(logits, dim = -1))
            masked_token_log_probs = log_softmax_probs[mask_index]
            masked_label_log_probs = log_softmax_probs[label_mask_index]
            potentials = torch.sum(masked_token_log_probs,dim=0)
            label_potentials = torch.sum(masked_label_log_probs,dim=0)
            return potentials, label_potentials

    def __eval(self,input_data):
        n_batches = len(input_data["input_ids"])//self.batch_size
        if len(input_data["input_ids"]) % self.batch_size != 0:
            n_batches += 1
        #print("Data size: {}".format(len(input_data["input_ids"])))
        #print("Batches: {}".format(n_batches))
        potentials = None
        label_potentials = None
        for i in range(n_batches):
            input = {}
            for k,v in input_data.items():
                input[k] = v[i*self.batch_size:min((i+1)*self.batch_size,len(v))]
                input[k] = torch.tensor(input[k]).to(self.device)
            if self.__is_t5():
                self.__convert_to_t5_inputs(input)
            pot, lab_pot = self.__eval_batch(input)
            #torch.cuda.empty_cache()
            if potentials == None :
                potentials = pot
            else:
                potentials = potentials + pot
            
            if label_potentials == None :
                label_potentials = lab_pot
            else:
                label_potentials = label_potentials + lab_pot
            
        return potentials, label_potentials
    '''
    def get_max_prob_tokens(self,n_tokens,masked_data,unlabeled_masked_data):
        input = copy.deepcopy(masked_data)
        #print(masked_data)
        #print(unlabeled_masked_data)
        for k, v in input.items():
            input[k] = v+unlabeled_masked_data[k]
            input[k] = torch.tensor(input[k]).to(self.device)
        mask_index = torch.where(input["input_ids"] == self.mask_token_id)
        input["input_ids"][torch.where(input["input_ids"] == self.place_holder_id)] = self.mask_token_id
        output = self.model(**input)
        logits = output.logits
        log_softmax_probs = torch.log(F.softmax(logits, dim = -1))
        half = log_softmax_probs.size(0)//2
        masked_token_log_probs = log_softmax_probs[mask_index]
        potentials = torch.sum(masked_token_log_probs[:half],dim=0)-torch.sum(masked_token_log_probs[half:],dim=0)
        top_k = torch.topk(potentials.squeeze(), n_tokens).indices
        return top_k.tolist()
    '''

    def __filter_label_bias(self,n_tokens,indices,label_ids):
        labels = []
        for lab in label_ids:
            labels.append(self.tokenizer.decode(lab).strip())
        ans = []
        for tok in indices:
            if isinstance(self.model,BertForMaskedLM) and tok <= self.tokenizer.mask_token_id:
                continue
            if tok in [self.tokenizer.mask_token_id,self.tokenizer.pad_token_id]:
                continue
            surf = self.tokenizer.decode(tok).strip().lower()
            skip = False
            for lab in labels:
                lower_lab = lab.lower()
                if lower_lab in surf or lower_lab.startswith(surf) or lower_lab.endswith(surf):
                    skip = True
                    break
            if not skip:
                ans.append(tok)
            if len(ans) >= n_tokens:
                break
        return ans

    def get_max_prob_tokens(self,n_tokens,masked_data,unlabeled_masked_data,label_ids, bias):
        token_potentials, _ = self.__eval(masked_data)
        unlabel_potentials, label_potentials = self.__eval(unlabeled_masked_data)
        potentials = token_potentials - unlabel_potentials + bias.to(self.device)
        top_k = torch.topk(potentials.squeeze(), n_tokens).indices
        top_k = top_k.tolist()

        return self.__filter_label_bias(n_tokens,top_k,label_ids), label_potentials, token_potentials, unlabel_potentials

    def sample_tokens(self,n_tokens,masked_data,unlabeled_masked_data,label_ids, bias):
        token_potentials, _ = self.__eval(masked_data)
        unlabel_potentials, label_potentials = self.__eval(unlabeled_masked_data)
        potentials = token_potentials - unlabel_potentials + bias.to(self.device)
        potentials = potentials.squeeze()
        #potentials[:self.tokenizer.cls_token_id] = -32768
        top_k = torch.topk(potentials, n_tokens).indices
        #filtered_potentials = (-32768*torch.ones(potentials.size())).to(self.device)
        #filtered_potentials[top_k] = potentials[top_k]
        probs = F.softmax(potentials[top_k], dim = -1)
        k_indices = torch.multinomial(probs, n_tokens)
        k_tokens = top_k[k_indices].tolist()

        return self.__filter_label_bias(n_tokens,k_tokens,label_ids), label_potentials, token_potentials, unlabel_potentials
        

class FitnessModelWrapper:

    ModelPool = {}

    def __init__(self,model_name_or_instance,custom_metric=None,batch_size=128,device=None,max_length=128):
        if isinstance(model_name_or_instance,str):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_instance,use_fast=False)
            if 't5' in model_name_or_instance:
                self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_instance)
            elif 'gpt' in model_name_or_instance:
                self.model = AutoModelForCausalLM.from_pretrained(model_name_or_instance)
            else:
                self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_instance)
                if 'roberta' in model_name_or_instance:
                    self.model.lm_head.bias = torch.nn.parameter.Parameter(torch.zeros(self.tokenizer.vocab_size))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_instance.tokenizer_name)
            self.model = model_name_or_instance
        
        if 'mask_token' not in self.tokenizer.special_tokens_map:
            #if isinstance(self.model,T5ForConditionalGeneration):
            #    self.tokenizer.add_special_tokens({'mask_token':'[mask]'})
            #else:
            self.tokenizer.add_special_tokens({'mask_token':'<extra_id_0>'})
        
        self.mask_token_id = self.tokenizer.mask_token_id
        self.device = device if device else torch.device('cpu')
        self.batch_size = batch_size
        self.max_length = max_length
        self.model.to(self.device)
        self.model.eval()
        self.offload_model = self.model
        self.offload_device = self.device
        self.metric = custom_metric
    
        if self.__is_gpt2():
            self.tokenizer.pad_token = self.tokenizer.eos_token
            #self.tokenizer.mask_token = self.tokenizer.pad_token
    
    def set_custom_metric(self,custom_metric):
        self.metric = custom_metric

    def __is_t5(self):
        return isinstance(self.model,T5ForConditionalGeneration)
    
    def __is_gpt2(self):
        return isinstance(self.model,GPT2LMHeadModel)

    def set_device(self,device_id):
        device = torch.device("cuda:{}".format(device_id)) 
        if device_id not in FitnessModelWrapper.ModelPool.keys():
            model = copy.deepcopy(self.offload_model)
            model.to(device)
            model.eval()
            FitnessModelWrapper.ModelPool[device_id] = model
        self.model = FitnessModelWrapper.ModelPool[device_id]
        self.device = device

    def offload(self):
        self.model = self.offload_model
        self.device = self.offload_device

    def __batch_eval(self,unlabeled_data,labels,label_ids):
        input = {}
        for k, v in unlabeled_data.items():
            input[k] = torch.tensor(v).to(self.device)
        
        if self.__is_t5():
            mask_index = torch.where(input["decode_input_ids"] == self.mask_token_id)
        elif self.__is_gpt2():
            end_of_sentence = torch.sum(input["attention_mask"],dim=-1).long() - 1
            mask_index = (
                torch.LongTensor([_ for _ in range(input["input_ids"].size(0))]).to(self.device),
                end_of_sentence,
            )
        else:
            mask_index = torch.where(input["input_ids"] == self.mask_token_id)

        with torch.no_grad():
            output = self.model(**input)
        logits = output.logits
        predicts = logits[mask_index]
        pred_log_probs = torch.log(F.softmax(predicts, dim = -1))
        
        if len(predicts) != len(labels):
            print("{} predicts vs. {} labels".format(len(predicts),len(labels)),flush=True)
            import pickle  
            with open('error.pkl','wb') as fp:
                pickle.dump(unlabeled_data,fp)
            for i in range(len(unlabeled_data['input_ids'])):
                print(unlabeled_data['input_ids'][i],flush=True)
        results = []
        preds = []
        fitness = []
        label_log_probs = []
        for i in range(len(labels)):
            if labels[i] != None:
                true_label_id = label_ids.index(labels[i])
            else:
                true_label_id = None
            pred_logits = predicts[i][label_ids]
            renorm_probs = torch.exp(pred_logits)/torch.sum(torch.exp(pred_logits))
            pred_id = torch.argmax(pred_logits)
            preds.append(label_ids[pred_id])
            results.append(int(pred_id==true_label_id))
                
            if labels[i] != None:
                #fitness.append(int(pred_id==true_label_id)*renorm_probs[true_label_id].item())
                fitness.append(renorm_probs[true_label_id].item())
                label_log_probs.append(pred_log_probs[i][labels[i]].item())
            else:
                fitness.append(None)
        return results, preds, fitness, label_log_probs

    def __eval(self,unlabeled_data,labels,label_ids):
        results = []
        preds = []
        fitness = []
        label_log_probs = []
        batchs = len(labels)//self.batch_size
        if len(labels) % self.batch_size != 0:
            batchs += 1
        #print("batch {}".format(batchs))
        for i in range(batchs):
            batch_data = {}
            for k, v in unlabeled_data.items():
                batch_data[k] = v[i*self.batch_size:min((i+1)*self.batch_size,len(labels))]
            batch_labels = labels[i*self.batch_size:min((i+1)*self.batch_size,len(labels))]
            batch_results, batch_preds, batch_fitness, batch_label_log_probs = self.__batch_eval(batch_data,batch_labels,label_ids)

            results += batch_results
            preds += batch_preds
            fitness += batch_fitness
            label_log_probs += batch_label_log_probs
        return results, preds, fitness, label_log_probs
    
    def fitness(self,individuals,data):
        eval_dataset,labels,label_ids = self.__construct_data(individuals,data)
        results, preds, fits, label_log_probs = self.__eval(eval_dataset,labels,label_ids)
        
        ind_chunk  = len(results)//len(individuals)
        
        for i in range(len(individuals)):
            acc = self.__accuracy(results[i*ind_chunk:(i+1)*ind_chunk])
            score = np.sum(fits[i*ind_chunk:(i+1)*ind_chunk])/ind_chunk
            individuals[i].fitness_score = score 

            if self.metric:
                metric_score = self.metric(labels,preds)
            individuals[i].accuracy = acc if not self.metric else metric_score

        return torch.tensor(label_log_probs)
        
    def label_log_probs(self,individuals,data):
        eval_dataset,labels,label_ids = self.__construct_data(individuals,data)
        _, _, _, label_log_probs = self.__eval(eval_dataset,labels,label_ids)
        return torch.tensor(label_log_probs)

    def predict(self,individual,data,label_tags):
        unlabeled_data, _ = add_prompts_to_data(data,individual.prompts,'mask',self.tokenizer.mask_token)
        if self.__is_t5():
            test_dataset = convert_to_t5_inputs(unlabeled_data,self.tokenizer,self.max_length)
        else:
            test_dataset = self.tokenizer(unlabeled_data,max_length=self.max_length,pad_to_max_length=True,truncation=True)
        #self.__collate(test_dataset)

        label_ids = []
        for lab in label_tags:
            lab_id = self.tokenizer.encode(' '+lab.strip(),add_special_tokens=False)[0]
            label_ids.append(lab_id)
        
        labels = [None]*len(data)

        _, preds, _, _ = self.__eval(test_dataset,labels,label_ids)
        return preds

    def __accuracy(self,results):
        #print(results)
        correct = np.sum(np.asarray(results))
        return correct/len(results)

    def __construct_data(self,individuals,data):
        unlabeled_data = []
        labels = []
        for ind in individuals:
            if self.__is_gpt2():
                text, lab = add_prompts_to_data(data,ind.prompts,False,'')
            else:
                text, lab = add_prompts_to_data(data,ind.prompts,'mask',self.tokenizer.mask_token)
            unlabeled_data += text
            labels += lab
        
        if self.__is_t5():
            eval_dataset = convert_to_t5_inputs(unlabeled_data,self.tokenizer,self.max_length)
        else:
            eval_dataset = self.tokenizer(unlabeled_data,max_length=self.max_length,pad_to_max_length=True,truncation=True)
        #self.__collate(eval_dataset)
        unique_labels = list(set(labels))
        label_ids = []
        for lab in unique_labels:
            lab_id = label_to_token_id(lab,self.tokenizer) #self.tokenizer.encode(' '+lab.strip(),add_special_tokens=False)[0]
            label_ids.append(lab_id)
        for i,lab in enumerate(labels):
            index = unique_labels.index(lab)
            labels[i] = label_ids[index]

        return eval_dataset, labels, label_ids
   