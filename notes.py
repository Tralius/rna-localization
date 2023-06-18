from GeneWrapper import Gene_Wrapper
import numpy as np

lower_bound = 0
upper_bound = np.inf
dataset = "apex-rip"


from collections import OrderedDict
import itertools

encoding_seq = OrderedDict([
    ('UNK', [0, 0, 0, 0]),
    ('A', [1, 0, 0, 0]),
    ('C', [0, 1, 0, 0]),
    ('G', [0, 0, 1, 0]),
    ('T', [0, 0, 0, 1]),
    ('N', [0.25, 0.25, 0.25, 0.25]),  # A or C or G or T
])

encoding_keys = list(encoding_seq.keys())


gene_data = Gene_Wrapper.seq_data_loader(False, dataset, lower_bound, upper_bound,permute = None)



template = [0] * 24  # dim([a,c,g,t]) * dim([f,t,i,h,m,s])
combined_encoding = OrderedDict()
combined_encoding['UNK'] = template
for i, (key_seq, key_ann) in enumerate(
        itertools.product(['A', 'C', 'G', 'T', 'N'], ['F', 'T', 'I', 'H', 'M', 'S'])):
    tmp = template.copy()
    if key_seq == 'N':
        for n in ['A', 'C', 'G', 'T']:
            tmp[np.nonzero(combined_encoding[n + key_ann])[0][0]] = 0.25
        combined_encoding[key_seq + key_ann] = tmp
    else:
        tmp[i] = 1  # normal one-hot encoding as it is...
        combined_encoding[key_seq + key_ann] = tmp
encoding_keys = list(combined_encoding.keys())
encoding_vectors = np.array(list(combined_encoding.values()))

