
def main():


    # Defining a few parameters
    max_len = 4000
    nb_filters = 32
    filters_length = 10
    pooling_size = 3
    lstm_units = 32
    lower_bound = 0
    upper_bound = 4000
    nb_classes = 9 # because we have 9 localisations
    batch_size = 256



    # Initializing test set
    np.random.seed(3)
    data_org = pd.read_csv('~/Downloads/final_data.csv')
    test_data = data_org.sample(frac=0.1)
    train_data = data_org.drop(test_data.index) # TODO: note: we also have to preprocess the test set similary
    # TODO: colab


    data_org

    sum_vec = train_data.iloc[:, :9].sum(axis=1)
    data2 = train_data.iloc[:, :9].divide(sum_vec, axis='index')
    train_data_no_struct = pd.concat([data2, train_data['seq']], axis=1)
    train_data_no_struct


    # One hot encode the 'seq' attribute of the above table
    mapping = {
        'A': 0,
        'C': 1,
        'G': 2,
        'T': 3
    }

    mapping_localisations = {
        'ERM':  0,
        'KDEL': 1,
        'LMA':  2,
        'MITO': 3,
        'NES':  4,
        'NIK':  5,
        'NLS':  6,
        'NUCP': 7,
        'OMM':  8
    }

    one_hot_encode_lam = lambda seq: to_categorical([mapping[x] for x in seq])
    data_one = train_data_no_struct['seq'].apply(one_hot_encode_lam)

    data_one

    # Now just injecting this modified 'seq' back into the pandas frame
    data_one_no_struct =  pd.concat([train_data_no_struct.iloc[:, :9], data_one], axis=1)

    data_one_no_struct


    # Additional ordinal encoding of the 'seq' attribute

    gene_data = train_data['seq']

    def label_dist(dist):
        # TODO: what is this
        assert (len(dist) == 4)
        return np.array(dist) / np.sum(dist)

    encoding_seq = OrderedDict([
        ('UNK', [0, 0, 0, 0]),
        ('A', [1, 0, 0, 0]),
        ('C', [0, 1, 0, 0]),
        ('G', [0, 0, 1, 0]),
        ('T', [0, 0, 0, 1]),
        ('N', [0.25, 0.25, 0.25, 0.25]),  # A or C or G or T
    ])

    encoding_keys = list(encoding_seq.keys())
    seq_encoding_vectors = np.array(list(encoding_seq.values()))
    encoding_vectors = seq_encoding_vectors

    X = pad_sequences([[encoding_keys.index(c) for c in gene] for gene in gene_data],
                        maxlen=max_len,
                        dtype=np.int8, value=encoding_keys.index('UNK'))  # , truncating='post')

    y = data_one_no_struct[mapping_localisations.keys()].values

    # y = np.array([label_dist(gene.dist) for gene in gene_data])

    #template = [0] * 24  # TODO: why 24? # dim([a,c,g,t]) * dim([f,t,i,h,m,s])
    #combined_encoding = OrderedDict()
    #combined_encoding['UNK'] = template
    #for i, (key_seq, key_ann) in enumerate(
    #        itertools.product(['A', 'C', 'G', 'T', 'N'], ['F', 'T', 'I', 'H', 'M', 'S'])):
    #    tmp = template.copy()
    #    if key_seq == 'N':
    #        for n in ['A', 'C', 'G', 'T']:
    #            tmp[np.nonzero(combined_encoding[n + key_ann])[0][0]] = 0.25
    #        combined_encoding[key_seq + key_ann] = tmp
    #    else:
    #        tmp[i] = 1  # normal one-hot encoding as it is...
    #        combined_encoding[key_seq + key_ann] = tmp
    #encoding_keys = list(combined_encoding.keys())
    #encoding_vectors = np.array(list(combined_encoding.values()))
    #encoding_vectors
    ##print(len(encoding_vectors))
    encoding_keys





    # Splitting for 5fold

    folds_total = 5

    kf = KFold(n_splits=folds_total, shuffle=True, random_state=1234)
    folds = kf.split(X, y)

    # folds now contains a list of lists. Each sublist contains all the indices for the pandas data entries to be used in the respective fold

    # Import NN
    from RNAtracker import RNATracker

    # TODO: do a normal CNN


    # Set paths for model output
    try:
        os.makedirs('~/Downloads/model_outputs')
    except Exception as e:
        print(str(e))

    model_output_folder = '~/Downloads/model_outputs'

    # TODO: with understand what we have to predict, we can allocate X and y
    # Also: the kwargsvalues are hyperparameters of which we will select default values from the RNAtracker repo
    epochs = 10

    for i, (train_indices, test_indices) in enumerate(folds):
        print('Evaluating KFolds {}/{}'.format(str(i + 1), str(folds_total)))
        model = RNATracker(max_len, nb_classes, model_output_folder, kfold_index=i)
        # model.build_model(nb_filters=kwargs['nb_filters'], filters_length=kwargs['filters_length'],
        #                          pooling_size=kwargs['pooling_size'], lstm_units=kwargs['lstm_units'],
        #                          embedding_vec=encoding_vectors)

        model.build_model_advanced_masking(nb_filters=nb_filters,
                                           filters_length=filters_length,
                                           pooling_size=pooling_size,
                                           lstm_units=lstm_units,
                                           embedding_vec=encoding_vectors)

        model.train(X[train_indices], y[train_indices], batch_size, epochs)

        model.evaluate(X[test_indices], y[test_indices], dataset)

        K.clear_session()

if __name__ == '__main__':
    main()


