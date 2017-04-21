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

# system
flags.DEFINE_integer('num_gpus', 8, 'the number of GPUs in the system')

# data
# TODO: mini dataset
flags.DEFINE_string ('mini_data_dir',   'mini_data',   'data directory. Should contain train.txt/valid.txt/test.txt with input data')
flags.DEFINE_string ('data_dir',        'data',   'data directory. Should contain train.txt/valid.txt/test.txt with input data')
flags.DEFINE_string ('population_dir',  'population', 'evolution history, information for generations')

# model params
flags.DEFINE_integer('char_embed_size', 15,                             'dimensionality of character embeddings')
flags.DEFINE_float  ('dropout',         0.5,                            'dropout. 0 = no dropout')
flags.DEFINE_integer('highway_layers',  2,                              'number of highway layers')

# evolution configuration
flags.DEFINE_integer('num_winners',             5, 'number of winners of each generation')
flags.DEFINE_integer('population_size',         30, 'number of individuals of each generation')
flags.DEFINE_integer('epoch',                   30, 'number of individuals of each generation')
flags.DEFINE_float  ('learning_threshold',      1.0, 'similarity threshold for teacher selection')
flags.DEFINE_float  ('prob_mutation_struct',    0.1, 'probability of mutation for individual structures')
flags.DEFINE_float  ('prob_mutation_param',     0.1, 'probability of mutation for individual parameters')
flags.DEFINE_integer('if_train_winner',         0, '1-train the winner; 0-do not train')

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
                cnn_layer={
                    "filter_type_1":[1, 50],
                    "filter_type_2":[2, 100],
                    "filter_type_3":[3, 150],
                    "filter_type_4":[4, 200],
                    "filter_type_5":[5, 200],
                    "filter_type_6":[6, 200],
                    "filter_type_7":[7, 200]},
                rnn_layers={
                    "layer_1":[650],
                    "layer_2":[650]},
                char_embed_size=FLAGS.char_embed_size
                dropout=FLAGS.dropout):

        self._individual_dir = FLAGS.population_dir + "/individual_%d" % self._id_number
        if not os.path.exists(self._individual_dir):
            os.mkdir(self._individual_dir)

        # TODO: generate individual seed
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
                        char_embed_size = char_embed_size,
                        dropout = dropout,
                        )

        # create model
        self._gpu_id = self._id_number % FLAGS.num_gpus
        self._graph = tf.Graph()
        self._model, self._valid_model, self._saver = self.create_graph()

        # initialize model
        with tf.Session(graph=self._graph, config=tf.ConfigProto(log_device_placement=True)) as session:
            tf.global_variables_initializer().run()
            session.run(self._model.clear_char_embedding_padding)
            print('Created and initialized fresh individual_%d. Size: %d' % (self._id_number, self._model.model_size()))
            self._summary_writer = tf.summary.FileWriter(self._individual_dir, graph=session.graph)
            session.run(
                tf.assign(self._model.learning_rate, FLAGS.learning_rate),
            )


    @classmethod
    def create_graph(self):
        with self._graph.as_default():
            # TODO: configure GPU
            with tf.device('/gpu:%d' % self._gpu_id):
                initializer = tf.random_uniform_initializer(-FLAGS.param_init, FLAGS.param_init)
                with tf.variable_scope("individual_%d" % self._id_number, initializer=initializer):
                    my_model = model.individual_graph(
                                            char_vocab_size=char_vocab.size,
                                            word_vocab_size=word_vocab.size,
                                            char_embed_size=self._knowledge.char_embed_size,
                                            batch_size=FLAGS.batch_size,
                                            max_word_length=FLAGS.max_word_length,
                                            num_unroll_steps=FLAGS.num_unroll_steps,
                                            num_highway_layers=2,
                                            cnn_layer=self._cnn_layer,
                                            rnn_layers=self._rnn_layers,
                                            dropout=self._knowledge.dropout)
                    my_model.update(model.loss_graph(my_model.logits, FLAGS.batch_size, FLAGS.num_unroll_steps))
                    my_model.update(model.training_graph(train_model.loss * FLAGS.num_unroll_steps, FLAGS.learning_rate, FLAGS.max_grad_norm))

                saver = tf.train.Saver()

                with tf.variable_scope("individual_%d" % self._id_number, reuse=True):
                    # TODO:
                    valid_model = model.individual_graph(
                                            char_vocab_size=char_vocab.size,
                                            word_vocab_size=word_vocab.size,
                                            char_embed_size=self._knowledge.char_embed_size,
                                            batch_size=FLAGS.batch_size,
                                            max_word_length=FLAGS.max_word_length,
                                            num_unroll_steps=FLAGS.num_unroll_steps,
                                            num_highway_layers=2,
                                            cnn_layer=self._cnn_layer,
                                            rnn_layers=self._rnn_layers,
                                        dropout=self._knowledge.dropout)
                valid_model.update(model.loss_graph(valid_model.logits, FLAGS.batch_size, FLAGS.num_unroll_steps))

        return my_model, valid_model, saver

    def update_graph(self):
        with self._graph.as_default():
            initializer = tf.random_uniform_initializer(-FLAGS.param_init, FLAGS.param_init)
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
        self._knowledge.char_embed_size = FLAGS.char_embed_size + np.random.randint(-FLAGS.char_embed_size, FLAGS.char_embed_size)
        self._knowledge.dropout = np.random.uniform()

    def mutation(self):
        for layer_type in self._layers:
        self.mutation_struct()
        self.mutation_param()

    # train on mini-dataset
    def fitness(self):
        word_vocab, char_vocab, word_tensors, char_tensors, max_word_length = load_data(FLAGS.mini_data_dir, FLAGS.max_word_length, eos=FLAGS.EOS)
        train_reader = DataReader(word_tensors['train'], char_tensors['train'], FLAGS.batch_size, FLAGS.num_unroll_steps)
        valid_reader = DataReader(word_tensors['valid'], char_tensors['valid'], FLAGS.batch_size, FLAGS.num_unroll_steps)

        fitness = 0
        with tf.Session(graph=self._graph, config=tf.ConfigProto(log_device_placement=True)) as session:
            with tf.variable_scope("individual_%d" % self._id_number, initializer=initializer):
                self.update_graph()
        return fitness

    # experience vector
    def experience(self):
        # TODO: how to model individual experience
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


class Population:
    def __init__(self,
                num_winners = 5,
                population_size = 30):

        self._num_winners = num_winners
        self._population_size = population_size

        # Individuals
        self._population = list()
        for i in range(self._population_size):
            self._population.append(self.generate(i))

    @classmethod
    def generate(self, id_number):
        individual = Individual(id_number=id_number)
        return individual

    def select(self):
        self._population.sort(key=lambda individual:individual.fitness(), reverse=True)
        winners = self._population[:self._num_winners]
        return winners

    def similarity(self, individual_1, individual_2):
        return np.linalg.norm(individual_1.experience() - individual_2.experience())

    def find_teacher(self, leaner):
        sim = FLAGS.learning_threshold
        teacher_id = -1
        for candidate_id in range(self._num_winners):
            cur_sim = self.similarity(self._population[candidate_id], leaner)
            if cur_sim > sim:
                sim = cur_sim
                teacher_id = candidate_id
        return teacher_id

    def evolve(self):
        self.select()
        for loser_id in range(self._num_winners, self._population_size):
            teacher_id = find_teacher(self._population[loser_id])
            if teacher_id >= 0:
                self._population[loser_id].learn(self._population[teacher_id].teach())
            else:
                self._population[loser_id].mutation()
            self._population[loser_id].update_graph()

    def final_winner(self):
        return self.select()[0]


def main(_):

    if not os.path.exists(FLAGS.population_dir):
        os.mkdir(FLAGS.population_dir)
        print('Created population history directory', FLAGS.population_dir)

    np.random.seed(seed=FLAGS.seed)

    # initialize population
    population = Population(num_winners=FLAGS.num_winners,
                            population_size=FLAGS.population)

    # perform evolution
    for epoch in range(FLAGS.epoch):
        population.evolve()

    # get result
    result = population.final_winner()
    result.save_model()

    # TODO: fully train
    if FLAGS.if_train_winner == 1:
        result.train()


if __name__ == '__main__':
    tf.app.run()
