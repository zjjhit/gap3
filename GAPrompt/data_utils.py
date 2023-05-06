import datasets

LABEL_POSITION = -1

def SET_LABEL_POSITION(pos):
    global LABEL_POSITION
    LABEL_POSITION = pos

def GET_LABEL_POSITION():
    global LABEL_POSITION
    return LABEL_POSITION

def load_raw_data(task_name,split,data_files):
    if data_files:
        raw_data = datasets.load_dataset(task_name, data_files=data_files, split=split)
    else:
        if isinstance(task_name,str):
            raw_data = datasets.load_dataset(task_name, split=split)
            if task_name in ['snli']:
                raw_data = raw_data.filter(lambda example: example['label'] in [0, 1, 2])
        elif isinstance(task_name,list) and len(task_name)==2 :
            raw_data = datasets.load_dataset(task_name[0], task_name[1], split=split)
    return raw_data    


def load_data_as_text_chunks(raw_dataset,keys,mapping=None,data_size_for_debug=None):
    #raw_dataset = load_dataset(task_name,dataset_name,split=split_name,cache_dir=os.path.join('.','data_cache',task_name),keep_in_memory=True)
    
    if data_size_for_debug != None and isinstance(data_size_for_debug,int):
        raw_dataset = raw_dataset[:data_size_for_debug]
    
    lists = []
    for k in keys:
        lists.append(raw_dataset[k])
    ans = []
    for i in range(len(lists[0])):
        example = []
        for x in lists:
            s = x[i]
            if mapping and isinstance(s,int):
                s = mapping[s]
            example.append(s)
        ans.append(example)
    return ans

'''
def add_prompts_to_data(data,prompts,with_labels=True,mask_token='[MASK]'):
    text = []
    labels = []
    for example in data:
        line = ''
        if isinstance(with_labels,bool) and with_labels:
            chunk = example[-1] 
        elif isinstance(with_labels,str) and with_labels=='mask':
            chunk = mask_token
        else:
            chunk = ''
        labels.append(example[-1].strip())
        
        i = -2
        while True:
            if len(prompts)+i+1 >= 0:
                prompt = prompts[i+1]
                
            else:
                prompt = ''
            if prompt:
                line = prompt+' '+chunk+' '+line
            else:
                line = chunk+' '+line
            if len(example)+i >= 0:
                chunk = example[i].strip()
                i -= 1
            else:
                break
        text.append(line.strip())
    return text,labels
'''

def add_prompts_to_data(data,prompts,with_labels=True,mask_token='[MASK]'):
    text = []
    labels = []
    global LABEL_POSITION
    for example in data:
        line = ''
        for i, chunk in enumerate(example):
            data_chunk = chunk
            if i == LABEL_POSITION:
                if isinstance(with_labels,str) and with_labels == 'mask':
                    data_chunk = mask_token
                if isinstance(with_labels,bool) and not with_labels:
                    data_chunk = ''

            prompt_chunk = ''
            if i < len(prompts) and prompts[i] != None:
                prompt_chunk = prompts[i]
            
            if prompt_chunk:
                line += ' ' + prompt_chunk
            if data_chunk:
                line += ' ' + data_chunk
        text.append(line.strip())
        labels.append(example[LABEL_POSITION])
    return text,labels


def add_prompt_ids_to_data(data,prompts,with_labels=True,mask_token='[MASK]'):
    text = []
    labels = []
    global LABEL_POSITION
    for example in data:
        line = ''
        for i, chunk in enumerate(example):
            data_chunk = chunk
            if i == LABEL_POSITION:
                if isinstance(with_labels,str) and with_labels == 'mask':
                    data_chunk = mask_token
                if isinstance(with_labels,bool) and not with_labels:
                    data_chunk = ''

            prompt_chunk = ''
            if i < len(prompts) and prompts[i] != None:
                prompt_chunk = prompts[i]
            
            if prompt_chunk:
                line += ' ' + prompt_chunk
            if data_chunk:
                line += ' ' + data_chunk
        text.append(line.strip())
        labels.append(example[LABEL_POSITION])
    return text,labels


def tokenize_data_chunks(data,tokenizer):
    tokenized_data = []
    for example in data:
        text = example[0]
        for i in range(1,len(example)):
            text += ' '+tokenizer.sep_token+' '+example[i]
        #print(text)
        token_ids = tokenizer.encode(text,add_special_tokens=False)
        #print(token_ids)
        one_example = []
        chunk = []
        for tok in token_ids:
            if tok == tokenizer.sep_token_id:
                one_example.append(chunk)
                chunk = []
            else:
                chunk.append(tok)
        if chunk:
            one_example.append(chunk)
        tokenized_data.append(one_example)
    return tokenized_data

def add_prompt_ids_to_data(data,
        prompts_ids,
        with_labels=True,
        cls_token_id=None,
        sep_token_id=None,
        pad_token_id=0,
        mask_token_id=None,
        max_length=512
    ):

    global LABEL_POSITION

    if cls_token_id != None:
        max_length -= 1
    if sep_token_id != None:
        max_length -= 1
    
    max_example_length = 0

    new_data = []
    labels = []
    for example in data:
        line = []
        labels.append(example[LABEL_POSITION][0])
        
        for i, chunk in enumerate(example):
            data_chunk = chunk
            if i == LABEL_POSITION:
                if isinstance(with_labels,str) and with_labels == 'mask':
                    data_chunk = [mask_token_id]
                if isinstance(with_labels,bool) and not with_labels:
                    data_chunk = []

            prompt_chunk = []
            if i < len(prompts_ids) and prompts_ids[i] != None:
                prompt_chunk = prompts_ids[i]
            
            line += prompt_chunk + data_chunk
            
        #truncation
        if len(line) > max_length:
            if LABEL_POSITION == len(prompts_ids)-1:
                del line[max_length-1:-1]
            else:
                line = line[:max_length]
        if cls_token_id != None: 
            line = [cls_token_id]+line
        if sep_token_id != None:
            line = line+[sep_token_id]
        if len(line) > max_example_length:
            max_example_length = len(line)
        new_data.append(line)

    ans = {'input_ids':[],'token_type_ids':[],'attention_mask':[]}
    for example in new_data:
        input_id = example
        attention_mask = [1]*len(input_id)
        content_len = len(input_id)
        if content_len < max_example_length:
            input_id += [pad_token_id]*(max_example_length-content_len)
            attention_mask += [0]*(max_example_length-content_len)
        token_type_id = [0]*max_example_length
        ans['input_ids'].append(input_id)
        ans['token_type_ids'].append(token_type_id)
        ans['attention_mask'].append(attention_mask)
    return ans, labels
