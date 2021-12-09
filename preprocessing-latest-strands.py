#!/usr/bin/env python
# coding: utf-8

# In[16]:


import nltk
import pandas as pd
import pickle
import inflection as inf
import string
import time
import sys
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()


# In[17]:


# Preprocessing output.txt files to contain method name and respective strand level vectors in a dictionary
def preprocess_text_file(output_file):
    s = time.time()    
    key=0
    data_dict ={}
    count = 0
    count_cpp = 0
    count_null_vectors = 0
    with open(output_file, "r") as lines:
        # read line by line
        for line in lines:
            if '.cpp' in line or '.cc' in line:
                count_cpp+=1
                continue
            #seperate file name and method name+vector
            # file_name:dsi_runtime_resume  -0.179542   0.381802...| 0.172 0.128 ....   
            name_seprator = line.split(":")
            # get the index of first tab, first tab is where method_name and vectors are seperated,
            # name_seprator[0] ==> file_name 
            # name_seprator[1] ==> method_name + vectors
            name_encoder_seprator_idx = name_seprator[1].find('\t')
            # till : will be method name so store it in method_name variable
            method_name = name_seprator[1][:name_encoder_seprator_idx].split('(')[0]
            # store all vectors in vector variable after :
            vectors = name_seprator[1][name_encoder_seprator_idx+1:]
            # strands vector are seperated by |, so split by | and store all the vectors in strands_vectors
    #             strands_vectors = vectors.split('|')
    #             # each element in vector can be seprated by \t 
    #             strands_vectors = [strand.split('\t') for strand in strands_vectors]
    #             #vector is converted to numpy array....
    #             final_vector = [np.array(strands) for strands in strands_vectors[:-1]]
            # data is stored as index = ['dsi_runtime_resume', [list of strands vectors]]
            strands_vectors = vectors.split('|')
# Length 1 strands vectors are ['\n'] where no vectors are generated for a function
            if len(strands_vectors)>1:
                count+=1
            # each element in vector can be seprated by \t
                temp_strands = []
                for strand in strands_vectors:
                    temp_vector = []
                    for element in strand.split('\t'):
                        if element.replace('.','',1).replace('-','',1).isdigit():
                            temp_vector.append(float(element))
                    temp_strands.append(temp_vector)
                    temp_strands = list(filter(None, temp_strands))
                data_dict[key] = [method_name, temp_strands]
                key+=1
    print('Time Taken to preprocess text file: {:.2f}s'.format(time.time()-s))
    print('Number of cpp enteries in file: {}'.format(count_cpp))
    print('Number of null vector enteries: {}'.format(count_null_vectors))
    # Both should be same in number
    print('Number of method names: {}'.format(len(data_dict.keys())))
    print('Count value: {}'.format(count))
#     print('-----------------------------------------------------------------------')
    return data_dict


# In[18]:


def preprocess_method_names_for_trie(data_dict): # list(data_dict.keys())
    s=time.time()
    preprocessed_data_dict = {}
    token_list = []
    for i in range(len(data_dict.keys())):
        method_name = data_dict[i][0]
        method_strand_vectors = data_dict[i][1]
        preprocessed_tokens = []

    # 1. Replacing camel case with snake case (underscore). Convertion to lower case also occurs by default.
        token_list = inf.underscore(method_name).split('_')
        for token in token_list:
            if token != '':
    # 2. Splitting using '_' and removing numbers from method names.
                token = ''.join(char for char in token if char.isalpha())
    # 3. Lemmatizing tokens to convert to base form (ending -> end)
                token = lemmatizer.lemmatize(lemmatizer.lemmatize(token, 'n'),'v')
                preprocessed_tokens.append(token)
        preprocessed_data_dict[i] = [method_name, preprocessed_tokens, method_strand_vectors]
    print('Time Taken to preprocess_method names for trie: {:.2f}s'.format(time.time()-s))
#     print('-----------------------------------------------------------------------')

    return preprocessed_data_dict
# ''.join(dict_value)


# In[19]:


def modify_english_vocab():
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
#   english_vocab = set(w.lower() for w in words)
    filter_set = set(list(string.ascii_lowercase))
    filter_set.add('jean-pierre')
    filter_set.add('jean-christophe')
    acronym_set = set(['sys', 'info', 'min', 'max', 'int', 'str', 'crc',
                       'cmd', 'tty', 'cb', 'fft', 'idx', 'ttx', 'irq',
                       'init', 'dir', 'cpu', 'len', 'pixel', 'config', 
                       'iterator', 'runtime', 'multi', 'os', 'matmul',
                       'num', 'col', 'malloc', 'alloc', 'mem', 'cpy', 'cmp'])
    english_vocab = (english_vocab - filter_set).union(acronym_set)
    return english_vocab


# In[20]:


class TrieNode:
    def __init__(self): 
        self.children = [None]*26
        self.leafNode = False

class Trie: 
    def __init__(self): 
        self.root = self.getNode() 

    def getNode(self): 
        # Returns new trie node (initialized to NULLs) 
        return TrieNode() 

    def _charToIndex(self,ch):
        # Converts key current character into index 
        # use only 'a' through 'z' and lower case
        return ord(ch)-ord('a') 

    def insert(self,key): 
        node = self.root 
        for i in range(len(key)): 
            index = self._charToIndex(key[i]) 
            if not node.children[index]: 
                node.children[index] = self.getNode() 
            node = node.children[index]
        node.leafNode = True

    def search(self, key): 
        node = self.root
        for i in range(len(key)): 
            index = self._charToIndex(key[i])
#             print(index)
#             print(node.children[index])
            if not node.children[index]:
#                 print(False, i)
                return (False, i)
            node = node.children[index]
        return (node != None and node.leafNode, i)
    
    def splitter(self, key):
        wholeStringTraversed = False
        temp_list = []
        while(not wholeStringTraversed):
            (leafNode, pos) = self.search(key)
            if leafNode is True:
                temp_list.append(key)
                break
            elif leafNode is False and pos == 0:
                return None
            else:
                temp_leafNode = leafNode
                temp_pos = pos
                while(not temp_leafNode):
                    temp_key = key[:temp_pos]
                    temp_leafNode, temp_pos = self.search(temp_key)
                pos = temp_pos + 1
                temp_list.append(key[:pos])
                key = key[pos:]
        return temp_list


# In[21]:


def preprocess_data_with_trie(preprocessed_data_dict):
    s= time.time()
    count=0
    trie_preprocessed_data_dict = []
    trie = Trie()
    english_vocab = modify_english_vocab()
    for key in english_vocab:
        trie.insert(key)
    
    list_of_generic_names = ['name', 'main', 'function', 'func', 'algo', 'algorithm', 'test']

    for i in range(len(preprocessed_data_dict.keys())):
        temp_tokens = []
        method_name = preprocessed_data_dict[i][0]
        list_of_tokens = preprocessed_data_dict[i][1]
        method_strand_vectors = preprocessed_data_dict[i][2]

        for token in list_of_tokens:
            split_token = []
            try:
                split_token = trie.splitter(token)
            except:
                split_token = None
            if split_token is not None:
                for i in split_token:
                    if i not in ['sodium', 'kiss']:
                        temp_tokens.append(i)
            else:
                temp_tokens.append(split_token)

    # Removing one occurance on None      
        if None in temp_tokens:
            temp_tokens.remove(None)

        if temp_tokens.count(None)==0 and len(temp_tokens)>1:
            trie_preprocessed_data_dict.append([method_name, temp_tokens, method_strand_vectors])
            count+=1
        elif temp_tokens.count(None)==0 and len(temp_tokens)==1 and temp_tokens[0] not in list_of_generic_names:
            trie_preprocessed_data_dict.append([method_name, temp_tokens, method_strand_vectors])
            count+=1
        else:
            trie_preprocessed_data_dict.append([method_name, None, method_strand_vectors])


    #     size = len(temp_tokens)
    #     if size<=1:
    #         trie_preprocessed_data_dict[i] = [method_name, None, method_strand_vectors]
    #     elif size==2 and (None in temp_tokens):
    #         trie_preprocessed_data_dict[i] = [method_name, None, method_strand_vectors]
    #     elif temp_tokens.count(None)>1:
    #         trie_preprocessed_data_dict[i] = [method_name, None, method_strand_vectors]
    #     else:
    #         if None in temp_tokens:
    #             temp_tokens.remove(None)
    #     trie_preprocessed_data_dict.append([method_name, temp_tokens, method_strand_vectors])

    print('Number of total method names in preprocessed_data_dict for trie: {}'.format(len(preprocessed_data_dict.keys())))
    print('Number of method names in trie_preprocessed_data_dict: {}'.format(len(trie_preprocessed_data_dict)))
    print('Number of meaningful method names for dataset: {}'.format(count))
    print('Time Taken to preprocess method names after trie filter: {:.2f}s'.format(time.time()-s))
#     print('-----------------------------------------------------------------------')
    return trie_preprocessed_data_dict


# In[22]:


def generate_data_for_model(trie_preprocessed_data_dict):
    s=time.time()
    tokens = []
    for each_row in trie_preprocessed_data_dict:
        if each_row[1] is not None:
            for token in each_row[1]:
                tokens.append(token)
    print('Total number of tokens: {}'.format(len(tokens)))
    print('Total number of unique tokens: {}'.format(len(list(set(tokens)))))
    word2index = {'SOS':0, 'EOS':1, 'PAD':2, 'UNK': 3}
    c = 4
    for token in list(set(tokens)):
        word2index[token] = c
        c += 1
    print('Total number of keys(tokens) in word2index: {}'.format(len(word2index.keys())))
    # model input is a list of [[name tensor], [encoding tensor], str_actual_name]
    model_input = []
    for each_row in trie_preprocessed_data_dict:
        method_name = each_row[0]
        list_of_tokens = each_row[1]
        method_strand_vectors = each_row[2]
        if list_of_tokens is not None:
            list_of_indexes = []
            for token in list_of_tokens:
                list_of_indexes.append(word2index[token])
            model_input.append([list_of_indexes, method_strand_vectors, method_name, list_of_tokens])

    #             model_input.append([list_of_indexes, data_dict[orig_method_name], orig_method_name])
    #             model_input.append([list_of_indexes, orig_method_name, list_of_tokens])

    print('Time Taken to generate data for model: {:.2f}s'.format(time.time()-s))
    print('Enteries in model_input: {}'.format(len(model_input)))
#     print('-----------------------------------------------------------------------')
    return (model_input, word2index)


# In[25]:


def main():
    print('PreProcessing the text file ...')
    print('-----------------------------------------------------------------------')
# Change the name of file
    data_dict = preprocess_text_file(sys.argv[1])
    print('-----------------------------------------------------------------------')
    print('Contents of preprocessed text file:')
    c=0
    for key, value in data_dict.items():
        if c<5:
            print(key, value[0], len(value[1]))
            c+=1
    print('-----------------------------------------------------------------------')
    preprocessed_data_dict = preprocess_method_names_for_trie(data_dict)
    print('-----------------------------------------------------------------------')
    print('PreProcessed method names before trie:')
    c = 0
    for key, value in preprocessed_data_dict.items():
        if c<5:
            print(key, value[0], value[1], len(value[2]))
            c+=1
    print('-----------------------------------------------------------------------')
    trie_preprocessed_data_dict = preprocess_data_with_trie(preprocessed_data_dict)
    print('-----------------------------------------------------------------------')
    #returning splitted_names (orig_name, tokenized name)
    print('PreProcessed method names after trie filter:')
    c=0
    for row in trie_preprocessed_data_dict:
        if c<20:
            print(row[0], row[1], len(row[2]))
            c+=1
    print('-----------------------------------------------------------------------')
    model_input, word2index = generate_data_for_model(trie_preprocessed_data_dict)
    print('-----------------------------------------------------------------------')
    print('Content of model_input:')
    c=0
    for lis in model_input:
        if c<10:
            print(lis[0], len(lis[1]),  lis[2], lis[3])
            c+=1
    c=0
    for lis in model_input:
        if len(lis[1]) == 1 and lis[1] is []:
            c+=1
    print('length 1 null enteries in model_input: {}'.format(c))
    print('-----------------------------------------------------------------------')    
    model_pkl_file = sys.argv[1].split('.')[0] + '_model_input.pkl'
    w2index_pkl_file = sys.argv[1].split('.')[0] + '_word2index.pkl'
    f1 = open(model_pkl_file, "wb")
    pickle.dump(model_input, f1)
    print('{} is generated.'.format(model_pkl_file))
    f2 = open(w2index_pkl_file, "wb")
    pickle.dump(word2index, f2)
    print('{} is generated.'.format(w2index_pkl_file))


# In[ ]:


if __name__ == '__main__':
    main()

