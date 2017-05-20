from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import copy
import logging
import threading

import numpy as np
import tensorflow as tf
from scipy.spatial import distance as dist

import model
from data_reader import load_data, load_mini_data, DataReader


flags = tf.flags

# system
flags.DEFINE_integer('num_cpus', 4, 'number of CPUs on the system')
flags.DEFINE_integer('num_gpus', 1, 'number of GPUs on the system')

# data
flags.DEFINE_integer('num_partitions',  300,  'total number of partitions for fitness training')
flags.DEFINE_string ('data_dir',        'data',   'data directory. Should contain train.txt/valid.txt/test.txt with input data')
flags.DEFINE_string ('population_dir',  'population', 'evolution history, information for generations')

# model params
flags.DEFINE_integer('char_embed_size', 15,                             'dimensionality of character embeddings')
flags.DEFINE_float  ('dropout',         0.5,                            'dropout. 0 = no dropout')
flags.DEFINE_integer('highway_layers',  2,                              'number of highway layers')

# evolution configuration
flags.DEFINE_integer('num_winners',             1, 'number of winners of each generation')
flags.DEFINE_integer('population_size',         3, 'number of individuals of each generation')
flags.DEFINE_integer('num_touraments',          1, 'number of tourament rounds of each generation')
flags.DEFINE_integer('max_evo_epochs',          20, 'max number of evolution iterations')
flags.DEFINE_float  ('neighborhood_radius',     10, 'similarity threshold for teacher selection')
flags.DEFINE_float  ('prob_mutation_struct',    0.1, 'probability of mutation for individual structures')
flags.DEFINE_float  ('prob_mutation_param',     0.1, 'probability of mutation for individual parameters')
flags.DEFINE_integer('max_cnn_filter_types',    21, 'max number of cnn filter types')
flags.DEFINE_integer('min_cnn_filter_types',    5, 'max number of cnn filter types')
flags.DEFINE_integer('max_cnn_type_filters',    300, 'max number of cnn filters for a specific type')
flags.DEFINE_integer('min_cnn_type_filters',    50, 'max number of cnn filters for a specific type')
flags.DEFINE_integer('max_rnn_layers',          5, 'max number of rnn layers')
flags.DEFINE_integer('min_rnn_layers',          1, 'max number of rnn layers')
flags.DEFINE_integer('max_rnn_layer_cells',     1000, 'max number of rnn layer cells')
flags.DEFINE_integer('min_rnn_layer_cells',     100, 'max number of rnn layer cells')
flags.DEFINE_integer('if_train_winner',         0, '1-train the winner; 0-do not train')
flags.DEFINE_integer('local_std_rate',          0.2, '')

# optimization
flags.DEFINE_float  ('learning_rate_decay', 0.5,  'learning rate decay')
flags.DEFINE_float  ('learning_rate',       1.0,  'starting learning rate')
flags.DEFINE_float  ('decay_when',          1.0,  'decay if validation perplexity does not improve by more than this much')
flags.DEFINE_float  ('param_init',          0.05, 'initialize parameters at')
flags.DEFINE_integer('num_unroll_steps',    35,   'number of timesteps to unroll for')
flags.DEFINE_integer('batch_size',          5,   'number of sequences to train on in parallel')
flags.DEFINE_integer('max_epochs',          1,   'number of full passes through the training data')
flags.DEFINE_float  ('max_grad_norm',       5.0,  'normalize gradients at')
flags.DEFINE_integer('max_word_length',     21,   'maximum word length')

# bookkeeping
flags.DEFINE_integer('seed',           3435, 'random number generator seed')
flags.DEFINE_integer('print_every',    25,    'how often to print current loss')
flags.DEFINE_string ('EOS',            '+',  '<EOS> symbol. should be a single unused character (like +) for PTB and blank for others')

FLAGS = flags.FLAGS


class adict(dict):
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self


class Individual:
    def __init__(self,
                id_number,
                cnn_layer={
                    "1":[1, 50],
                    "2":[2, 100],
                    "3":[3, 150],
                    "4":[4, 200],
                    "5":[5, 200],
                    "6":[6, 200],
                    "7":[7, 200]},
                rnn_layers={
                    "1":[650],
                    "2":[650]},
                char_embed_size=FLAGS.char_embed_size,
                dropout=FLAGS.dropout):

        self._seed = np.random.seed(id_number * 13)
        self._id_number = id_number
        self._max_word_length = 0
        self._fitness = 0
        self._score = 0
        self._loss = 0
        self._model_size = 0

        self._individual_dir = FLAGS.population_dir + "/individual_%d" % self._id_number
        if not os.path.exists(self._individual_dir):
            os.mkdir(self._individual_dir)

        # layer_i:          { filter_type_1, ..., filter_type_n } 
        # filter_type_j:    [ size, number ]
        # size, number: integer
        self._cnn_layer = cnn_layer

        # _rnn_layers:  { layer_1, ..., layer_n }
        # layer_i:      [ size ]
        self._rnn_layers = rnn_layers

        # encode network structure
        self._knowledge = adict(
                        char_embed_size = [char_embed_size],
                        dropout = [dropout],
                        struct_exp = [self.encode_structure()]
                        )

        self.show_knowledge()

        # create model
        self._gpu_id = self._id_number % FLAGS.num_gpus
        self._graph = None

    def show_knowledge(self, exp=-1):
        # print("[EVOLUTION] Individual_%d EXP_%d embed size:" % (self._id_number, exp), self._knowledge.char_embed_size[exp])
        # print("[EVOLUTION] Individual_%d EXP_%d dropout:" % (self._id_number, exp), self._knowledge.dropout[exp])
        print("[EVOLUTION] Individual_%d EXP_%d Conv:\n" % (self._id_number, exp), self._knowledge.struct_exp[exp][0])
        print("[EVOLUTION] Individual_%d EXP_%d Recu:\n" % (self._id_number, exp), self._knowledge.struct_exp[exp][1])

    def get_exp(self):
        return self._knowledge.struct_exp

    def get_fitness(self):
        return self._fitness

    def get_score(self):
        return self._score

    def get_loss(self):
        return self._loss

    def get_model_size(self):
        return self._model_size

    def encode_structure(self):
        struct_cnn = np.zeros([FLAGS.max_cnn_filter_types], dtype=np.int32)
        struct_rnn = np.zeros([FLAGS.max_rnn_layers], dtype=np.int32)
        # vector for CNN
        for filter_type in self._cnn_layer.values():
            struct_cnn[filter_type[0]-1] = filter_type[1]
        # vector for RNN
        for layer_i, num_units in self._rnn_layers.items():
            struct_rnn[int(layer_i)-1] = num_units[0]
        structure = [copy.deepcopy(struct_cnn), copy.deepcopy(struct_rnn)]
        return structure

    def decode_structure(self, struct):
        cnn_layer = {}
        rnn_layers = {}
        struct_cnn = struct[0]
        struct_rnn = struct[1]
        for filter_type in range(struct_cnn.shape[0]):
            if struct_cnn[filter_type] > 0:
                cnn_layer['%d' % (filter_type+1)] = [filter_type+1, int(struct_cnn[filter_type])]
        for layer in range(struct_rnn.shape[0]):
            if struct_rnn[layer] > 0:
                rnn_layers['%d' % (layer+1)] = [int(struct_rnn[layer])]
        self._cnn_layer = copy.deepcopy(cnn_layer)
        self._rnn_layers = copy.deepcopy(rnn_layers)
        return self._cnn_layer, self._rnn_layers

    def experience(self, struct):
        self._knowledge.struct_exp.append(struct) 

    def update_graph(self, word_vocab, char_vocab, max_word_length, epoch):
        # TODO: release previous graph?
        self._graph = tf.Graph()
        # self._graph.clear_collection(self._graph.get_all_collection_keys())
        with self._graph.as_default():
            tf.set_random_seed(self._seed)
            with tf.device('/gpu:%d' % self._gpu_id):
                initializer = tf.random_uniform_initializer(-FLAGS.param_init, FLAGS.param_init)
                # TODO: other way to rebuild graph?
                reuse = None
                # if epoch > 0:
                #     reuse = True
                with tf.variable_scope("Epoch_%d_Individual_%d" % (epoch, self._id_number), initializer=initializer, reuse=reuse):
                    my_model = model.individual_graph(
                                            char_vocab_size=char_vocab.size,
                                            word_vocab_size=word_vocab.size,
                                            char_embed_size=self._knowledge.char_embed_size[-1],
                                            batch_size=FLAGS.batch_size,
                                            max_word_length=max_word_length,
                                            num_unroll_steps=FLAGS.num_unroll_steps,
                                            num_highway_layers=2,
                                            cnn_layer=self._cnn_layer,
                                            rnn_layers=self._rnn_layers,
                                            dropout=self._knowledge.dropout[-1])
                    my_model.update(model.loss_graph(my_model.logits, FLAGS.batch_size, FLAGS.num_unroll_steps))
                    my_model.update(model.training_graph(my_model.loss * FLAGS.num_unroll_steps, FLAGS.learning_rate, FLAGS.max_grad_norm))

                saver = tf.train.Saver(max_to_keep=1)

                with tf.variable_scope("Epoch_%d_Individual_%d" % (epoch, self._id_number), reuse=True):
                    valid_model = model.individual_graph(
                                            char_vocab_size=char_vocab.size,
                                            word_vocab_size=word_vocab.size,
                                            char_embed_size=self._knowledge.char_embed_size[-1],
                                            batch_size=FLAGS.batch_size,
                                            max_word_length=max_word_length,
                                            num_unroll_steps=FLAGS.num_unroll_steps,
                                            num_highway_layers=2,
                                            cnn_layer=self._cnn_layer,
                                            rnn_layers=self._rnn_layers,
                                            dropout=self._knowledge.dropout[-1])
                    valid_model.update(model.loss_graph(valid_model.logits, FLAGS.batch_size, FLAGS.num_unroll_steps))
        return my_model, valid_model, saver

    def mutation_struct(self, local=True):

        # mutate cnn
        # mutate number of filters
        for filter_type in self._cnn_layer.values():
            if local:
                num_cnn_type_filters = int(np.random.normal(filter_type[1], filter_type[1] * FLAGS.local_std_rate))
                if num_cnn_type_filters > FLAGS.max_cnn_type_filters:
                    num_cnn_type_filters = FLAGS.max_cnn_type_filters 
                elif num_cnn_type_filters < FLAGS.min_cnn_type_filters:
                    num_cnn_type_filters = FLAGS.min_cnn_type_filters 
                # add / remove filters
                filter_type[1] = num_cnn_type_filters
            else:
                # add / remove filters
                filter_type[1] = np.random.randint(FLAGS.min_cnn_type_filters, FLAGS.max_cnn_type_filters+1)
        # mutate filter types
        if local:
            num_cnn_filter_types = int(np.random.normal(len(self._cnn_layer), len(self._cnn_layer) * FLAGS.local_std_rate))
            if num_cnn_filter_types > FLAGS.max_cnn_filter_types:
                num_cnn_filter_types = FLAGS.max_cnn_filter_types
            elif num_cnn_filter_types < FLAGS.min_cnn_filter_types:
                num_cnn_filter_types = FLAGS.min_cnn_filter_types
            num_var_cnn_filter_types = num_cnn_filter_types - len(self._cnn_layer) 
        else:
            num_cnn_filter_types = np.random.randint(FLAGS.min_cnn_filter_types, FLAGS.max_cnn_filter_types+1)
            num_var_cnn_filter_types = num_cnn_filter_types - len(self._cnn_layer) 
        # add filter type
        if num_var_cnn_filter_types > 0: # and len(self._cnn_layer) + num_var_cnn_filter_types <= min(FLAGS.max_cnn_filter_types, self._max_word_length):
            available_types = list(set(filter_type_i for filter_type_i in range(1, self._max_word_length+1))-set(filter_type[0] for filter_type in self._cnn_layer.values()))
            for i in range(num_var_cnn_filter_types):
                new_type = np.random.choice(available_types)
                available_types.remove(new_type)
                num_new_type = np.random.randint(FLAGS.min_cnn_type_filters, FLAGS.max_cnn_type_filters+1)
                self._cnn_layer['%d' % new_type] = [new_type, num_new_type]
        # remove filter type
        elif num_var_cnn_filter_types < 0: # and len(self._cnn_layer) + num_var_cnn_filter_types > 0:
            existed_types = [filter_type[0] for filter_type in self._cnn_layer.values()]
            for i in range(-num_var_cnn_filter_types):
                selected_type = np.random.choice(existed_types)
                existed_types.remove(selected_type)
                self._cnn_layer.pop('%d' % selected_type)
            
        # mutate rnn
        # mutate number of units
        for layer in self._rnn_layers.values():
            if local:
                num_rnn_layer_units = int(np.random.normal(len(self._rnn_layers), len(self._rnn_layers) * FLAGS.local_std_rate))
                if num_rnn_layer_units > FLAGS.max_rnn_layer_cells:
                    num_rnn_layer_units = FLAGS.max_rnn_layer_cells
                elif num_rnn_layer_units < FLAGS.min_rnn_layer_cells:
                    num_rnn_layer_units = FLAGS.min_rnn_layer_cells
            else:
                num_rnn_layer_units = np.random.randint(FLAGS.min_rnn_layer_cells, FLAGS.max_rnn_layer_cells+1)
            layer[0] = num_rnn_layer_units
        # mutate number of rnn layers 
        if local:
            num_rnn_layers = int(np.random.normal(len(self._rnn_layers), len(self._rnn_layers) * FLAGS.local_std_rate))
            if num_rnn_layers > FLAGS.max_rnn_layers:
                num_rnn_layers = FLAGS.max_rnn_layers
            elif num_rnn_layers < FLAGS.min_rnn_layers:
                num_rnn_layers = FLAGS.min_rnn_layers
            num_var_rnn_layers = num_rnn_layers - len(self._rnn_layers)
        else:
            num_rnn_layers = np.random.randint(FLAGS.min_rnn_layers, FLAGS.max_rnn_layers+1)
            num_var_rnn_layers = num_rnn_layers - len(self._rnn_layers)
        # add rnn layer
        if num_var_rnn_layers > 0: # and len(self._rnn_layers) + num_var_rnn_layers <= FLAGS.max_rnn_layers:
            for j in range(num_var_rnn_layers):
                for i in range(1, FLAGS.max_rnn_layers+1):
                    if not self._rnn_layers.get('%d' % i):
                        num_units = np.random.randint(FLAGS.min_rnn_layer_cells, FLAGS.max_rnn_layer_cells+1)
                        self._rnn_layers['%d' % i] = [num_units]
                        break
        # remove rnn layer
        elif num_var_rnn_layers < 0: # and len(self._rnn_layers) + num_var_rnn_layers > 0:
            for i in range(-num_var_rnn_layers):
                existed_layers = [layer for layer in self._rnn_layers.keys()]
                selected_layer = np.random.choice(existed_layers)
                existed_layers.remove(selected_layer)
                self._rnn_layers.pop(selected_layer)

        # refresh knowledge
        structure = self.encode_structure()
        self.experience(structure)

    def mutation_param(self):
        self._knowledge.char_embed_size.append(FLAGS.char_embed_size + np.random.randint(-(FLAGS.char_embed_size-1), FLAGS.char_embed_size))
        self._knowledge.dropout.append(np.random.uniform(0, 0.9))

    def mutation(self, local=True):
        # deprecated
        # self.mutation_param()
        self.mutation_struct()

    # train on mini-dataset
    def loss(self, word_vocab, char_vocab, word_tensors, char_tensors, max_word_length, evo_epoch, tourament):
        if 0 == tourament:
            self._model, self._valid_model, self._saver = self.update_graph(word_vocab, char_vocab, max_word_length, evo_epoch)

        self._max_word_length = max_word_length

        train_reader = DataReader(word_tensors['train'], char_tensors['train'], FLAGS.batch_size, FLAGS.num_unroll_steps)
        valid_reader = DataReader(word_tensors['valid'], char_tensors['valid'], FLAGS.batch_size, FLAGS.num_unroll_steps)

        with tf.Session(graph=self._graph, config=tf.ConfigProto(allow_soft_placement=True, inter_op_parallelism_threads=FLAGS.num_cpus)) as session:
            # initialize model
            tf.global_variables_initializer().run()
            session.run(self._model.clear_char_embedding_padding)
            if 0 == tourament:
                self._model_size = model.model_size()
            # print('[EVOLUTION] Epoch_%d Individual_%d. Size: %d' % (evo_epoch, self._id_number, self._model_size))
            # self._summary_writer = tf.summary.FileWriter(self._individual_dir, graph=session.graph)
            session.run(
                tf.assign(self._model.learning_rate, FLAGS.learning_rate),
            )
            # scout train
            best_valid_loss = None
            rnn_state = session.run(self._model.initial_rnn_state)
            # train a mini
            # epoch_start_time = time.time()
            # avg_train_loss = 0.0
            count = 0
            for x, y in train_reader.iter():
                count += 1
                start_time = time.time()
                loss, _, rnn_state, gradient_norm, step, _ = session.run([
                    self._model.loss,
                    self._model.train_op,
                    self._model.final_rnn_state,
                    self._model.global_norm,
                    self._model.global_step,
                    self._model.clear_char_embedding_padding
                ], {
                    self._model.input  : x,
                    self._model.targets: y,
                    self._model.initial_rnn_state: rnn_state
                })
                # avg_train_loss += 0.05 * (loss - avg_train_loss)
                # time_elapsed = time.time() - start_time
            # print('training time:', time.time()-epoch_start_time)
            # time to evaluate
            avg_valid_loss = 0.0
            count = 0
            rnn_state = session.run(self._valid_model.initial_rnn_state)
            for x, y in valid_reader.iter():
                count += 1
                start_time = time.time()
                loss, rnn_state = session.run([
                    self._valid_model.loss,
                    self._valid_model.final_rnn_state
                ], {
                    self._valid_model.input  : x,
                    self._valid_model.targets: y,
                    self._valid_model.initial_rnn_state: rnn_state,
                })
                avg_valid_loss += loss / valid_reader.length
            # print("train loss = %6.8f, perplexity = %6.8f" % (avg_train_loss, np.exp(avg_train_loss)))
            # print("validation loss = %6.8f, perplexity = %6.8f" % (avg_valid_loss, np.exp(avg_valid_loss)))
            # save_as = '%s/epoch%03d_%.4f.model' % (self._individual_dir, evo_epoch, avg_valid_loss)
            # save_as = '%s/epoch%03d.model' % (self._individual_dir, evo_epoch)
            # self._saver.save(session, save_as)
            # print('Saved model', save_as)
            ''' decide if need to decay learning rate '''
            if best_valid_loss is not None and np.exp(avg_valid_loss) > np.exp(best_valid_loss) - FLAGS.decay_when:
                print('validation perplexity did not improve enough, decay learning rate')
                current_learning_rate = session.run(self._model.learning_rate)
                print('learning rate was:', current_learning_rate)
                current_learning_rate *= FLAGS.learning_rate_decay
                if current_learning_rate < 1.e-5:
                    print('learning rate too small - stopping now')
                    self._fitness = best_valid_loss
                    return self._fitness
                session.run(self._model.learning_rate.assign(current_learning_rate))
                print('new learning rate is:', current_learning_rate)
            else:
                best_valid_loss = avg_valid_loss

        if tourament >= FLAGS.num_touraments:
            tf.reset_default_graph()

        self._loss = best_valid_loss
        return self._loss

    def absorb(self, struct_exp, cnn_probs, rnn_probs):
        cnn_last = struct_exp[-1][0]
        rnn_last = struct_exp[-1][1]
        beneficial_exp = []
        cnn_selected = []
        rnn_selected = []
        for i in range(int(FLAGS.max_cnn_filter_types * 0.3)):
            prob = 0
            for cnn_i in range(len(cnn_last)):
                if cnn_last[cnn_i] > 0 and cnn_probs[cnn_i] > prob and (cnn_i not in cnn_selected):
                    prob = cnn_probs[cnn_i]
                    cnn_selected.append(cnn_i)
                    if cnn_last[cnn_i] == 0 and self._knowledge.struct_exp[-1][0][cnn_i] > 0 and len(np.nonzero(self._knowledge.struct_exp[-1][0])[0]) == 1:
                        print('')
                    else:
                        self._knowledge.struct_exp[-1][0][cnn_i] = cnn_last[cnn_i]
        for i in range(int(FLAGS.max_rnn_layers * 0.4)):
            prob = 0
            for rnn_i in range(len(rnn_last)):
                if rnn_last[rnn_i] > 0 and rnn_probs[rnn_i] > prob and (rnn_i not in rnn_selected):
                    prob = rnn_probs[rnn_i]
                    rnn_selected.append(rnn_i)
                    if rnn_last[rnn_i] == 0 and self._knowledge.struct_exp[-1][1][rnn_i] > 0 and len(np.nonzero(self._knowledge.struct_exp[-1][1])[0]) == 1:
                        print('')
                    else:
                        self._knowledge.struct_exp[-1][1][rnn_i] = rnn_last[rnn_i]
        beneficial_exp.append([cnn_selected, rnn_selected])
        self.decode_structure(self._knowledge.struct_exp[-1])
        self.experience(self._knowledge.struct_exp[-1])
        return beneficial_exp

    # encode and return evolution knowledge
    def teach(self):
        return self._knowledge
        
    # decode and absorb evolution knowledge
    def learn(self, knowledge, cnn_probs, rnn_probs):
        self._knowledge.char_embed_size.append(knowledge.char_embed_size[-1])
        self._knowledge.dropout.append(knowledge.dropout[-1])
        return self.absorb(knowledge.struct_exp, cnn_probs, rnn_probs)

    def train(self):
        return


class Population:
    def __init__(self,
                num_winners = 3,
                population_size = 10):

        self._epoch = 0

        self._num_winners = num_winners
        self._population_size = population_size

        self._cnn_filter_stat = np.zeros([FLAGS.max_cnn_filter_types], dtype='int32')
        self._rnn_layers_stat = np.zeros([FLAGS.max_rnn_layers], dtype='int32')
        self._cnn_filter_wins = np.zeros([FLAGS.max_cnn_filter_types], dtype='int32')
        self._rnn_layers_wins = np.zeros([FLAGS.max_rnn_layers], dtype='int32')
        self._cnn_probs = np.zeros([FLAGS.max_cnn_filter_types])
        self._rnn_probs = np.zeros([FLAGS.max_rnn_layers])

        # TODO: multi threads for multi GPUs
        self._threadings = []

        # Individuals
        self._population = list()
        for i in range(self._population_size):
            self._population.append(self.generate(i))
        print("[EVOLUTION] Initialized Population")

        # self._average_fitness = None

    ''' deprecated
    def average_fitness(self):
        self._average_fitness = 0 
        for individual in self._population:
            self._average_fitness += individual.get_fitness()
        self._average_fitness /= self._population_size
        return self._average_fitness
    '''

    # with Laplace correction
    def probability(self, i, wins, appears, total):
        return (wins[i]+1) / (appears[i]+total)

    def update_probs(self):
        # Laplace correction
        for i in range(len(self._cnn_probs)):
            self._cnn_probs[i] = self.probability(i, self._cnn_filter_wins, self._cnn_filter_stat, FLAGS.max_cnn_filter_types)
        for i in range(len(self._rnn_probs)):
            self._rnn_probs[i] = self.probability(i, self._rnn_layers_wins, self._rnn_layers_stat, FLAGS.max_rnn_layers)
        print("[EVOLUTION] cnn probs:", self._cnn_probs)
        print("[EVOLUTION] rnn probs:", self._rnn_probs)
        
    def update_struct_stat(self, cnn_filters, rnn_layers):
        for f in range(len(cnn_filters)):
            if cnn_filters[f] > 0:
                self._cnn_filter_stat[f] += 1
        for l in range(len(rnn_layers)):
            if rnn_layers[l] > 0:
                self._rnn_layers_stat[l] += 1

    def update_struct_wins(self, cnn_filters, rnn_layers):
        for f in range(len(cnn_filters)):
            if cnn_filters[f] > 0:
                self._cnn_filter_wins[f] += 1
        for l in range(len(rnn_layers)):
            if rnn_layers[l] > 0:
                self._rnn_layers_wins[l] += 1

    def final_winner(self):
        return self._population[:self._num_winners]

    def generate(self, id_number):
        cnn_layer = {}
        rnn_layers = {}
        # generate cnn layer
        # len: 5 - 21
        num_filter_types = np.random.randint(FLAGS.min_cnn_filter_types, FLAGS.max_cnn_filter_types+1)
        available_filter_types = [i for i in range(1, FLAGS.max_cnn_filter_types+1)]
        for i in range(num_filter_types):
            filter_type = np.random.choice(available_filter_types)
            available_filter_types.remove(filter_type)
            # num: 50 - 300
            num_type_filters = np.random.randint(FLAGS.min_cnn_type_filters, FLAGS.max_cnn_type_filters)
            cnn_layer['%d' % filter_type] = [filter_type, num_type_filters]
        # generate rnn layers
        # 1 - 5
        num_rnn_layers = np.random.randint(FLAGS.min_rnn_layers, FLAGS.max_rnn_layers+1)
        for layer in range(1, num_rnn_layers+1):
            # 100 - 1000
            num_units = np.random.randint(FLAGS.min_rnn_layer_cells, FLAGS.max_rnn_layer_cells+1)
            rnn_layers['%d' % layer] = [num_units]
        individual = Individual(id_number=id_number,
                                cnn_layer=cnn_layer,
                                rnn_layers=rnn_layers)
        print("[EVOLUTION] Generated Individual_%d" % id_number)
        return individual

    def select(self, epoch):
        self.update_probs()
        for individual in self._population:
            individual._score = self._population_size * FLAGS.num_touraments
        for t in range(FLAGS.num_touraments):
            arena = np.random.randint(1, FLAGS.num_partitions+1)
            ranking = self.tourament(epoch, t, arena)
            print("[EVOLUTION] fitness ranking:", [individual._id_number for individual in self._population], "fitness:", [individual.get_fitness() for individual in self._population])
            for rank in range(len(ranking)):
                ranking[rank]._score -= rank * ranking[rank].get_fitness()
        self._population.sort(key=lambda individual:individual.get_score(), reverse=True)
        winners = self._population[:self._num_winners]
        # update structure statistics
        for ind_rank in range(self._population_size):
            self.update_struct_stat(self._population[ind_rank]._knowledge.struct_exp[-1][0], self._population[ind_rank]._knowledge.struct_exp[-1][1])
            if ind_rank <= self._num_winners:
                self.update_struct_wins(self._population[ind_rank]._knowledge.struct_exp[-1][0], self._population[ind_rank]._knowledge.struct_exp[-1][1])
        return winners

    def fitness(self, loss, losses, model_size, model_sizes):
        reg_size = (model_size - np.min(model_sizes)) / np.std(model_sizes)
        reg_loss = (loss - np.min(losses)) / np.std(losses)
        fitness = 1.05 * reg_size + reg_loss
        return fitness

    def tourament(self, epoch, t, partition):
        word_vocab, char_vocab, word_tensors, char_tensors, max_word_length = load_mini_data(FLAGS.data_dir, FLAGS.max_word_length, eos=FLAGS.EOS, partition=partition, num_partitions=FLAGS.num_partitions)
        losses = []
        model_sizes = []
        for individual in self._population:
            losses.append(individual.loss(word_vocab, char_vocab, word_tensors, char_tensors, max_word_length, epoch, t))
            model_sizes.append(individual.get_model_size())
        for individual in self._population:
            loss = individual.get_loss()
            model_size = individual.get_model_size()
            individual._fitness = self.fitness(loss, losses, model_size, model_sizes)
            print("[EVOLUTION] Epoch_%d Tour_%d Individual_%d fitness: %f, loss: %f, size: %d" % (epoch, t, individual._id_number, individual.get_fitness(), loss, model_size))
        self._population.sort(key=lambda individual:individual.get_fitness())
        return self._population

    def seuclidean(self, x, y):
        assert len(x) == len(y), "x, y with different dimentionality."
        dim = len(x)
        V = np.array([np.var([x[i], y[i]]) for i in range(dim)])
        # correction for 0
        V[V<1] = 1
        # print(x)
        # print(y)
        # print(V)
        return dist.seuclidean(x, y, V)

    def distance(self, individual_1, individual_2):
        ''' history make no sense
        time_discount = 0.9
        dissim_cnn, dissim_rnn = 0, 0
        '''
        exp1 = individual_1.get_exp()
        exp2 = individual_2.get_exp()
        assert len(exp1) == len(exp2), ("Individuals' experience missed.", exp1, exp2)
        '''
        for struct1, struct2 in zip(exp1, exp2):
            dissim_cnn = (time_discount * dissim_cnn + np.linalg.norm(struct1[0] - struct2[0])) / self.dissim_norm(time_discount, len(exp1))
            dissim_rnn = (time_discount * dissim_rnn + np.linalg.norm(struct1[1] - struct2[1])) / self.dissim_norm(time_discount, len(exp1))
        return 100 / (dissim_cnn + dissim_rnn)
        '''
        struct_1 = np.concatenate((exp1[-1][0], exp1[-1][1]))
        struct_2 = np.concatenate((exp2[-1][0], exp2[-1][1]))
        return self.seuclidean(struct_1, struct_2)

    def dissim_norm(self, td, steps):
        if steps > 1:
            return 1 + td * self.dissim_norm(td, steps-1)
        else:
            return 1

    def find_teacher(self, loser):
        further_dist = 0
        teacher = None
        for candidate_rank in range(self._num_winners):
            distance = self.distance(loser, self._population[candidate_rank])
            print("[EVOLUTION] %d and %d distance: %f" % (loser._id_number, self._population[candidate_rank]._id_number, distance))
            if distance < FLAGS.neighborhood_radius:
                if distance > further_dist:
                    further_dist = distance 
                    teacher = self._population[candidate_rank]
                    print("[EVOLUTION] Loser_%d and Winner_%d distance: %f" % (loser._id_number, teacher._id_number, distance))
        return teacher

    def evolve(self, epoch):
        self._epoch = epoch
        self.select(epoch)
        # display ranking
        print("[EVOLUTION] score ranking:", [individual._id_number for individual in self._population], "score:", [individual.get_score() for individual in self._population])
        # average fitness
        # print("[EVOLUTION] average fitness:", np.average([individual.get_fitness() for individual in self._population]))
        for loser_rank in range(self._num_winners, self._population_size):
            loser = self._population[loser_rank]
            teacher = self.find_teacher(loser)
            if teacher != None:
                new_struct = loser.learn(teacher.teach(), self._cnn_probs, self._rnn_probs)
                print("[EVOLUTION] Individual_%d learn from Individual_%d: " % (loser._id_number, teacher._id_number))
                # print("[EVOLUTION] - char_embed_size: ", loser._knowledge.char_embed_size[-1])
                # print("[EVOLUTION] - dropout: ", loser._knowledge.dropout[-1])
                print("[EVOLUTION] - Conv: ", loser._cnn_layer)
                print("[EVOLUTION] - Recu: ", loser._rnn_layers)
                print("[EVOLUTION] - learned: ", new_struct)
            else:
                loser.mutation(local=True)
                print("[EVOLUTION] Individual_%d mutate:" % loser._id_number)
                # print("[EVOLUTION] - char_embed_size: ", loser._knowledge.char_embed_size[-1])
                # print("[EVOLUTION] - dropout: ", loser._knowledge.dropout[-1])
                print("[EVOLUTION] - Conv: ", loser._cnn_layer)
                print("[EVOLUTION] - Recu: ", loser._rnn_layers)

        winner_to_mutate = []
        for winner_1_rank in range(self._num_winners-1):
            for winner_2_rank in range(winner_1_rank, self._num_winners):
                winner_1 = self._population[winner_1_rank]
                winner_2 = self._population[winner_2_rank]
                if self.distance(winner_1, winner_2) <= FLAGS.neighborhood_radius:
                    if winner_1.get_fitness() > winner_2.get_fitness():
                        winner_to_mutate.append(winner_2_rank)
                    else:
                        winner_to_mutate.append(winner_1_rank)
        for winner_rank in range(self._num_winners):
            winner = self._population[winner_rank]
            if winner_rank in winner_to_mutate:
                winner.mutation(local=False)
            else:
                winner._knowledge.char_embed_size.append(winner._knowledge.char_embed_size[-1])
                winner._knowledge.dropout.append(winner._knowledge.dropout[-1])
                winner.experience(winner._knowledge.struct_exp[-1])
                winner.show_knowledge()

    def show_history(self, individual):
        for exp in range(self._epoch):
            individual.show_knowledge(exp)


def main(_):

    # TODO: more readable
    logging.basicConfig(filename='%s/history.log' % FLAGS.population_dir, format='%(levelname)s:%(message)s', level=logging.INFO)

    if not os.path.exists(FLAGS.population_dir):
        os.mkdir(FLAGS.population_dir)
        print('[EVOLUTION] Created population history directory', FLAGS.population_dir)

    np.random.seed(seed=int(time.time()))

    # initialize population
    population = Population(num_winners=FLAGS.num_winners,
                            population_size=FLAGS.population_size)

    # perform evolution
    for epoch in range(FLAGS.max_evo_epochs):
        print('==================================================================================================================================')
        print('[EVOLUTION] Epoch_%d started' % epoch)
        print('==================================================================================================================================')
        population.evolve(epoch)

    # get result
    winners = population.final_winner()
    for winner in winners:
        population.show_history(winner)

    # TODO: plot automatically

    # TODO: fully train
    if FLAGS.if_train_winner == 1:
        result.train()


if __name__ == '__main__':
    tf.app.run()
