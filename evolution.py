from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import copy
import numpy as np
import tensorflow as tf

import model
from data_reader import load_data, DataReader


flags = tf.flags

# data
flags.DEFINE_string('data_dir',    'data',   'data directory. Should contain train.txt/valid.txt/test.txt with input data')
flags.DEFINE_string ('evolve_dir', 'evo_history', 'evolution history, information for generations')

# model params
flags.DEFINE_integer('highway_layers',  2,                              'number of highway layers')

# evolution configuration
flags.DEFINE_integer('population', 10, 'number of individuals of each generation')
flags.DEFINE_integer('epoch', 30, 'number of individuals of each generation')
flags.DEFINE_integer('mini_batch_size', 5, 'size of mini-batch for fitness test')
flags.DEFINE_integer('mini_num_unroll_steps', 5, 'size of mini-timesteps for fitness test')
flags.DEFINE_float  ('prob_mutation_struct', 0.1, 'probability of mutation for individual structures')
flags.DEFINE_float  ('prob_mutation_param', 0.1, 'probability of mutation for individual parameters')

# optimization
flags.DEFINE_float  ('learning_rate_decay', 0.5,  'learning rate decay')
flags.DEFINE_float  ('learning_rate',       1.0,  'starting learning rate')
flags.DEFINE_float  ('decay_when',          1.0,  'decay if validation perplexity does not improve by more than this much')
flags.DEFINE_float  ('param_init',          0.05, 'initialize parameters at')
flags.DEFINE_integer('num_unroll_steps',    35,   'number of timesteps to unroll for')
flags.DEFINE_integer('batch_size',          20,   'number of sequences to train on in parallel')
flags.DEFINE_integer('max_epochs',          25,   'number of full passes through the training data')
flags.DEFINE_float  ('max_grad_norm',       5.0,  'normalize gradients at')
flags.DEFINE_integer('max_word_length',     65,   'maximum word length')

# bookkeeping
flags.DEFINE_integer('seed',           3435, 'random number generator seed')
flags.DEFINE_integer('print_every',    5,    'how often to print current loss')
flags.DEFINE_string ('EOS',            '+',  '<EOS> symbol. should be a single unused character (like +) for PTB and blank for others')

FLAGS = flags.FLAGS


class adict(dict, *av, **kav):
    def __init__(self):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self


class Individual:
    def __init__(self,
                id_number,
                cnn_layer,
                rnn_layers):
        # TODO(LEON): if multi cnn layers necessary?
        # _cnn_layers:      { layer_1, ..., layer_n }
        # layer_i:          { filter_type_1, ..., filter_type_n } 
        # filter_type_j:    [ size, number ]
        # size, number: integer
        # self._cnn_layers

        self._seed = np.random.seed(id_number * 13)
        self._id_number = id_number
        # layer_i:          { filter_type_1, ..., filter_type_n } 
        # filter_type_j:    [ size, number ]
        # size, number: integer
        self._cnn_layer = cnn_layer
        # _rnn_layers:  { layer_1, ..., layer_n }
        # layer_i:      [ size ]
        self._rnn_layers = rnn_layers

        self._knowledge = adict(
                        char_embed_size = 15,
                        dropout = 0.5,
                        )

        self._model, self._valid_model = self.create_graph()

        tf.global_variables_initializer().run()
        session.run(self._model.clear_char_embedding_padding)
        print('Created and initialized fresh model. Size:', model.model_size())

        summary_writer = tf.summary.FileWriter(FLAGS.evolve_dir, graph=session.graph)

        session.run(
            tf.assign(self._model.learning_rate, FLAGS.learning_rate),
        )


    @classmethod
    def create_graph(self):
        # TODO: at present, all individuals are initialized identically
        initializer = tf.random_uniform_initializer(-FLAGS.param_init, FLAGS.param_init)
        with tf.variable_scope("individual_%d" % self._id_number, initializer=initializer):
            my_model = model.inidividual_graph(
                                    char_vocab_size=char_vocab.size,
                                    word_vocab_size=word_vocab.size,
                                    self._knowledge.char_embed_size,
                                    batch_size=FLAGS.batch_size,

                                    self._knowledge.dropout
                                    )
            my_model.update(model.loss_graph(train_model.logits, FLAGS.batch_size, FLAGS.num_unroll_steps))
        with tf.variable_scope("individual_%d" % self._id_number, reuse=True):
            valid_model = 
            valid_model.update(model.loss_graph(valid_model.logits, FLAGS.batch_size, FLAGS.num_unroll_steps))

        return my_model, valid_model

    def update_graph(self):
        with tf.variable_scope("individual_%d" % self._id_number, initializer=initializer):


    def mutation_struct(self):
        # mutate cnn
        num_var_cnn_filters = np.random.randint(-1, 1)
        if num_var_cnn_filters > 0:
            for filter in self._cnn_layer:
                break
        elif num_var_cnn_filters < 0:
            if len(self._cnn_layer) > num_var_cnn_filters:
                break
            else:
                # len = 1
        # mutate rnn


    def mutation_param(self):
        # TODO: knowledge should be learned instead
        self._knowledge.char_embed_size = np.random.randint(20)
        self._knowledge.dropout = np.random.uniform()

    def mutation(self):
        for layer_type in self._layers:
        self.mutation_struct()
        self.mutation_param()

    # train for only one epoch
    # TODO: other evaluation method?
    #   solution 1: train a mini dataset
    def fitness(self):
        train_reader = DataReader(word_tensors['train'], char_tensors['train'], FLAGS.batch_size, FLAGS.num_unroll_steps)
        valid_reader = DataReader(word_tensors['valid'], char_tensors['valid'], FLAGS.batch_size, FLAGS.num_unroll_steps)
        with tf.variable_scope("Model", initializer=initializer):
            self.update_graph()

    # experience vector
    def experience(self):
        # TODO: how to model similar experience
        exp = np.array([1, 1])
        return exp

    # encode and return evolution knowledge
    def teach(self):
        return self._knowledge
        
    # decode and absorb evolution knowledge
    def learn(self, knowledge):
        self._knowledge = copy.deepcopy(knowledge)

    def save_model(self):


    def train(self):
        return


class Generation:
    def __init__(self,
                num_winners = 3,
                population_size = 10):
        self._num_winners = num_winners
        self._population_size = population_size
        # Individuals
        self._population = list()
        for i in range(self._population_size):
            self._population.append(self.generate())

    @classmethod
    def generate(self):
        individual = Individual()
        return individual

    def select(self):
        self._population.sort(key=lambda individual:individual.fitness(), reverse=True)
        winners = self._population[:self._num_winners]
        return winners

    def similarity(self, individual_1, individual_2):
        return np.linalg.norm(individual_1 - individual_2)

    def find_teacher(self, leaner):
        sim = 0
        for candidate_id in range(self._num_winners):
            cur_sim = self.similarity(self._population[candidate_id], leaner)
            if cur_sim > sim:
                sim = cur_sim
                teacher_id = candidate_id
        return teacher_id

    # select and generate
    def evolve(self):
        # selection
        self._population = self.select()
        for i in range(self._population_size - self._num_winners):
            # add new individual
            self._population.append(self.generate())
            # mutation
            self._population[-1].mutation()
            # learn knowledge (crossover)
            teacher_id = find_teacher(self._population[-1])
            self._population[-1].learn(self._population[teacher_id].teach())
        # mutation
        for i in range(self._num_winners):
            self._population[i].mutation()

    def final_winner(self):
        return self.select()[0]


if __name__ == '__main__':
    if not os.path.exists(FLAGS.evolve_dir):
        os.mkdir(FLAGS.evolve_dir)
        print('Created evolution history directory', FLAGS.evolve_dir)

    word_vocab, char_vocab, word_tensors, char_tensors, max_word_length = load_data(FLAGS.data_dir, FLAGS.max_word_length, eos=FLAGS.EOS)

    with tf.Graph().as_default(), tf.Session() as session:

        tf.set_random_seed(FLAGS.seed)
        np.random.seed(seed=FLAGS.seed)

        generation = Generation()
        for epoch in range(FLAGS.epoch):
            generation.evolve()
        result = generation.final_winner()
        result.save_model()
        if FLAGS.train_winner == 1:
            result.train()
