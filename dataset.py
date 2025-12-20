import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler
import lightning as L

import pandas as pd
import numpy as np

import random
import ast
import copy 
import os
import csv

def identify_tokens(sequence):

    token = []
    token_long = False
    tokening = ""
    for word in sequence:

        if (word != "<" and word !=">") and not token_long:
            token.append(word)
        elif word == "<":
            token_long = True
            tokening += word
        elif token_long and word != ">":
            tokening += word
        elif token_long and word == ">":
            token_long = False
            tokening += word
            token.append(tokening)
            tokening = ""

    return token

def one_hot_encode_sequence(sequence, token_dicts, seq_length):

    one_hot_encoded = np.zeros((len(token_dicts), seq_length))
    tokenized_sequence = identify_tokens(sequence)

    for i, token in enumerate(tokenized_sequence):

        one_hot_encoded[int(token_dicts[token])][i] = 1  


    for i  in range(len(tokenized_sequence), seq_length):

        one_hot_encoded[int(token_dicts['<blank>'])][i] = 1  

    return np.array(one_hot_encoded)

def decode_condition_vectors(condition_labels, conditions):
    species = []
    objects = []
    groups = []
    mic = []
    condition_label_copy = copy.deepcopy(condition_labels)
    for i, condition_label in enumerate(condition_label_copy[0:3]):
        condition_label_copy[i] = {v: k for k, v in condition_label.items()}

    for condition in conditions:


        species.append(condition_label_copy[0][np.where(condition[0:6] == 1)[0][0]])
        objects.append(f'{[condition_label_copy[1][i] for i in np.where(condition[6:11] == 1)[0]]}')
        groups.append(f'{[condition_label_copy[2][i] for i in np.where(condition[11:16] == 1)[0]]}')
        if np.where(condition[16:] == 1)[0][0] == 0:
            mic.append(f'{[0, condition_label_copy[3][np.where(condition[16:] == 1)[0][0]]]}')
        elif np.where(condition[16:] == 1)[0][0] == 9:
            value = condition_label_copy[3][np.where(condition[16:] == 1)[0][0]]
            mic.append(f'{[value,value * 5]}')
        elif np.where(condition[16:] == 1)[0].size != 0:
            mic.append(f'{[condition_label_copy[3][np.where(condition[16:] == 1)[0][0] - 1],condition_label_copy[3][np.where(condition[16:] == 1)[0][0]]]}')
        else:
            mic.append(f'{[9999,10000]}')

    df_dict = {'speices' :species, 
               'objects' :objects, 
               'groups' :groups, 
               'mic' :mic, 
               }
    df = pd.DataFrame(df_dict)
    return df

def find_occurrences(sequence, token):
    occurrences = []
    start = 0

    while True:
        start = sequence.find(token, start)
        if start == -1:
            break
        occurrences.append(start)
        start += len(token)  
    return occurrences

def decode_sequences(tokens, sequences, generate_sample=False):
    tokens_copy = copy.deepcopy(tokens)
    tokens_copy = {int(v): k for k, v in tokens.items()}
    final_sequence = []

    one_hot_encoded = np.zeros((sequences.shape[0],sequences.shape[1], sequences.shape[2]))
    for seq_ind, generated_sequence in enumerate(sequences):

        for i, j in enumerate(np.argmax(generated_sequence, axis = 0)):
            one_hot_encoded[seq_ind][j][i] = 1
    
    for i, sequence in enumerate(one_hot_encoded):
        sequence_list = []
        for row in sequence.T:
            if np.where(row == 1)[0].size == 1:
                sequence_list.append(tokens_copy[np.where(row == 1)[0][0]])
        final_sequence.append("".join(sequence_list))

    if generate_sample:

        cropped_sequence_right = []
        for i, sequence, in enumerate(final_sequence):
            index = find_occurrences(sequence, '<EOS>')
            if len(index) != 0:
                if sequence[index[0] + 5: index[0] + 10] == "<AMD>" and index[0] + 10 < sequences.shape[2]:
                    cropped_sequence_right.append(sequence[:index[0] + 10])
                elif sequence[index[0] + 5: index[0] + 13] == "<cblank>" and index[0] + 13 < sequences.shape[2]:
                    cropped_sequence_right.append(sequence[:index[0] + 13])
                else:
                    cropped_sequence_right.append(sequence[:index[0] + 5])
            else:
                cropped_sequence_right.append(f"{sequence}")


        return cropped_sequence_right
    else:
        return final_sequence
    
class AMPConditions:
    def __init__(self, df):
        self.df = df

        self.target_species = ['escherichia coli', 'pseudomonas aeruginosa', 'klebsiella pneumoniae',
                               'staphylococcus aureus', 'bacillus subtilis', 'staphylococcus epidermidis']        
        self.target_objects = ['LIPID BILAYER', 'DNA / RNA', 'CYTOPLASMIC PROTEIN', 'MEMBRANE PROTEIN', 'OTHER']
        self.target_groups  = ['GRAM-', 'GRAM+', 'MAMMALIAN CELL', 'FUNGUS', 'OTHER']        

        self.species_dict = {species_name: i  for i, species_name in enumerate(self.target_species)}
        self.groups_dict = {groups_name: i for i, groups_name in enumerate(self.target_groups)}
        self.objects_dict = {objects_name: i for i, objects_name in enumerate(self.target_objects)}
    
        self.tokens, self.tokens_dict = self.load_tokens()
        self.log_mean_mic, self.log_std_mic, self.mic_min, self.mic_max = self.bin_mic()

    def load_tokens(self, special_tokens=['<blank>', '<MASK>']):
        tokens = np.array(self.df['modified_sequence'].apply(identify_tokens))
        tokens = set(np.concatenate(tokens))

        tokens -= set(special_tokens)

        tokens = sorted(tokens)
        tokens = [special_tokens[0]] + tokens + [special_tokens[1]] 

        tokens_dict = {token: i for i, token in enumerate(tokens)}

        if os.path.exists("data/dict.csv"):
            with open("data/dict.csv") as csv_file:
                reader = csv.reader(csv_file)
                temp = {key: int(value) for key, value in reader}

            if len(set(temp.keys()) & set(tokens)) == len(tokens):
                tokens_dict = temp
        else:
            with open("data/dict.csv", "w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerows(tokens_dict.items())

            print("Created new data/dict.csv")

        return tokens, tokens_dict

    def bin_mic(self):
        categories, self.bin_edges = pd.qcut(self.df['MIC'], q=10, labels=False, retbins=True)
        self.df['MIC_category'] = categories

        # For Regressions
        self.df['MIC_norm'] = np.log(self.df['MIC'] + 1e-6) 
        log_mean_mic = self.df['MIC_norm'].mean()
        log_std_mic = self.df['MIC_norm'].std()
        self.df['MIC_norm'] = (self.df['MIC_norm'] - log_mean_mic) / log_std_mic
    
        mic_min = np.min(self.df['MIC_norm'])
        mic_max = np.max(self.df['MIC_norm'])

        return log_mean_mic, log_std_mic, mic_min, mic_max

class AMPDatasets(Dataset):
    def __init__(self, data_path = "data/", max_length = 64):

        # nterminus and cterminus
        self.max_length = max_length

        # load datasets
        self.dbaasp_df = pd.read_csv(os.path.join(data_path, "dbaasp.csv"), index_col=0)

        self.dbaasp_df['targetGroups'] = self.dbaasp_df['targetGroups'].apply(ast.literal_eval)
        self.dbaasp_df['targetObjects'] = self.dbaasp_df['targetObjects'].apply(ast.literal_eval)

        self.conditions = AMPConditions(self.dbaasp_df)

        # sequences one-hot encoding
        self.sequences = []
        self.condition = []

        for row in self.conditions.df.values:
            encoded_species = np.zeros(6)
            encoded_groups  = np.zeros(5)   
            encoded_objects = np.zeros(5)
            encoded_mic     = np.zeros(10)   

        
            # create binary encoding for conditions (species, group, target and mic)
            encoded_species[self.conditions.species_dict[row[0]]] = 1

            for target_group in row[2]:
                encoded_groups[self.conditions.groups_dict[target_group]] = 1

            for target_object in row[3]:
                encoded_objects[self.conditions.objects_dict[target_object]] = 1

            encoded_mic[row[5]] = 1

            self.sequences.append(one_hot_encode_sequence(row[1], self.conditions.tokens_dict, self.max_length))

            #####
            # AMPGAN is not using raw mic rather than one hot encoded MIC
            #####
            # self.condition.append(np.concatenate([encoded_species, encoded_groups, encoded_objects, [row[6]]]))
            
            self.condition.append(np.concatenate([encoded_species, encoded_groups, encoded_objects, encoded_mic]))


    def __len__(self):
        return len(self.dbaasp_df)

    def __getitem__(self, idx):
        sample = self.sequences[idx]
        condition = self.condition[idx]
        label = 1 
        return {
            "sequence": sample,
            "condition": condition,
            "label": label
        }

class NonAMPDatasets(Dataset):
    def __init__(self, data_path ="data/", max_length = 64, label = 0):
        self.max_length = max_length
        self.label_neg = label

        self.non_amp_df = pd.read_csv(os.path.join(data_path, "non_amps.csv"), index_col=0)
        self.non_amp_df['Sequence'] = self.non_amp_df['Sequence'].apply(lambda x:"<nblank><SOS>"+x+"<EOS><cblank>")
        
        conditions  = AMPConditions(pd.read_csv(os.path.join(data_path, "dbaasp.csv"), index_col=0))

        self.non_sequences = []
        self.non_conditions = []
        

        for row in self.non_amp_df.values:
            encoded_species = np.zeros(6)
            encoded_groups  = np.zeros(5)   
            encoded_objects = np.zeros(5)
            encoded_mic     = np.zeros(10)

            # random species
            np.random.seed(42)
            encoded_species[np.random.randint(0,6)] = 1

            # This is raw mic value
            # mic_value = np.random.uniform(
            # (np.log(150 + 1e-6) - conditions.log_mean_mic) / conditions.log_std_mic,
            # (np.log(conditions.df['MIC'].max() + 1e-6) - conditions.log_mean_mic) / conditions.log_std_mic
            # )
            
            # This is binarized mic value
            encoded_mic[np.random.randint(len(encoded_mic))] = 1

            self.non_sequences.append(one_hot_encode_sequence(row[0], conditions.tokens_dict, self.max_length))
            self.non_conditions.append(np.concatenate([encoded_species, 
                                                       encoded_groups, 
                                                       encoded_objects, 
                                                       encoded_mic]))

    def __len__(self):
        return len(self.non_sequences)

    def __getitem__(self, idx):
        sample = self.non_sequences[idx]
        condition = self.non_conditions[idx]
        label = self.label_neg
        return {
            "sequence": sample,
            "condition": condition,
            "label": label
        }

class BatchSampler(Sampler):
    def __init__(self, pos_indices, neg_indices, batch_size, pos_ratio=0.5):
        self.pos_indices = pos_indices
        self.neg_indices = neg_indices
        self.batch_size = batch_size
        self.pos_ratio = pos_ratio 
        self.pos_batch_size = int(batch_size * pos_ratio)
        self.neg_batch_size = batch_size - self.pos_batch_size

    def __iter__(self):
        pos_samples = random.sample(self.pos_indices, len(self.pos_indices))
        neg_samples = random.sample(self.neg_indices, len(self.neg_indices))
        
        neg_len = len(neg_samples)
        neg_ptr = 0

        for pos_ptr in range(0, len(pos_samples), self.pos_batch_size):
            pos_batch = pos_samples[pos_ptr:pos_ptr + self.pos_batch_size]

            if len(pos_batch) < self.pos_batch_size:
                break  

            if neg_ptr + self.neg_batch_size > neg_len:
                neg_samples = random.sample(self.neg_indices, len(self.neg_indices))
                neg_ptr = 0
            neg_batch = neg_samples[neg_ptr:neg_ptr + self.neg_batch_size]
            neg_ptr += self.neg_batch_size

            batch = pos_batch + neg_batch
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return len(self.pos_indices) // self.pos_batch_size
    
class AMPDatasetModule(L.LightningDataModule):
    def __init__(self, file_path="data/", max_length=64, batch_size=128, pos_ratio=0.5):
        super().__init__()
        self.file_path = file_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.pos_ratio = pos_ratio

    def setup(self, stage=None):
        self.amp_dataset = AMPDatasets(data_path=self.file_path, max_length=self.max_length)
        self.non_amp_dataset = NonAMPDatasets(data_path=self.file_path, max_length=self.max_length)
        self.full_dataset = ConcatDataset([self.amp_dataset, self.non_amp_dataset])
        self.token_dict = self.amp_dataset.conditions.tokens_dict
        self.pos_indices = list(range(len(self.amp_dataset)))
        self.neg_indices = [i + len(self.amp_dataset) for i in range(len(self.non_amp_dataset))]

        self.sampler = BatchSampler(
            self.pos_indices,
            self.neg_indices,
            batch_size=self.batch_size,
            pos_ratio=self.pos_ratio
        )

    def train_dataloader(self):
        return DataLoader(
            self.full_dataset,
            batch_sampler=self.sampler,
            num_workers=4, 
            pin_memory=True
        )

class SwissProtDataset(Dataset):
    def __init__(self, data_path = "data/", max_length=66):
        self.df = pd.read_csv(os.path.join(data_path, "swissprot.csv"))
        self.df['raw_sequence'] = self.df['sequence'].apply(lambda x: x.split(">")[1].split("<")[0])
        self.sequence_list = self.df['sequence']
        
        self.max_length = max_length

        self.tokens, self.tokens_dict = self.load_tokens()
        self.sequences = []


        for sequence in self.sequence_list:
            self.sequences.append(one_hot_encode_sequence(sequence, self.tokens_dict, max_length))


    def load_tokens(self, special_tokens=['<blank>', '<MASK>']):
        tokens = np.array(self.df['sequence'].apply(identify_tokens))
        tokens = set(np.concatenate(tokens))

        tokens -= set(special_tokens)

        tokens = sorted(tokens)
        tokens = [special_tokens[0]] + tokens + [special_tokens[1]] 

        tokens_dict = {token: i for i, token in enumerate(tokens)}

        if os.path.exists("data/tokens_swissprot.csv"):
            with open("data/tokens_swissprot.csv") as csv_file:
                reader = csv.reader(csv_file)
                temp = {key: int(value) for key, value in reader}

            if len(set(temp.keys()) & set(tokens)) == len(tokens):
                tokens_dict = temp
        else:
            with open("data/tokens_swissprot.csv", "w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerows(tokens_dict.items())

            print("Created new data/tokens_swissprot.csv")

        return tokens, tokens_dict

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]


        return {
            "sequence": seq,

        }

class SwissProtModule(L.LightningDataModule):
    def __init__(self, data_path="data/", max_length=66, batch_size=128):
        super().__init__()
        self.data_path = data_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.categorical_bin = categorical_bin
        
        
    def setup(self, stage=None):
        self.full_dataset = SwissProtDataset(data_path=self.data_path, 
                                             max_length=self.max_length)

    def train_dataloader(self):
        return DataLoader(
            self.full_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    def predict_dataloader(self):
        return DataLoader(
            self.full_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
