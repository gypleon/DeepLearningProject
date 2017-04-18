import os
import time
import numpy as np
import tensorflow as tf

import model


flags = tf.flags

# evolution configuration
flags.DEFINE_string ('evolve_dir', 'evo_history', 'evolution history, information for generations')
flags.DEFINE_integer('population', 10, 'number of individuals of each generation')
flags.DEFINE_integer('epoch', 30, 'number of individuals of each generation')
flags.DEFINE_integer('mini_batch_size', 5, 'size of mini-batch for fitness test')
flags.DEFINE_integer('mini_num_unroll_steps', 5, 'size of mini-timesteps for fitness test')
flags.DEFINE_float  ('prob_mutation_struct', 0.1, 'probability of mutation for individual structures')
flags.DEFINE_float  ('prob_mutation_param', 0.1, 'probability of mutation for individual parameters')

FLAGS = flags.FLAGS

class Individual:
    def __init__(self,
                cnn_layers,
                rnn_layers):
        # _cnn_layers:  { layer_1, ..., layer_n }
        # layer_i:      { filter_type_1, ..., filter_type_n } 
        # filter_type_j:     [ size, number ]
        # size, number: integer
        self._cnn_layers
        # _rnn_layers:  { layer_1, ..., layer_n }
        # layer_i:      { neuron_1, ..., neuron_j }
        self._rnn_layers

    @classmethod
    def create_graph():

    def mutation_struct(self, layer):

    def mutation_param(self, ):

    def mutation(self):
        for layers in self._cnn_layers:
            self.mutation_struct()
        sefl.mutation_param()

    # train for only one epoch
    # TODO: other evaluation method?
    #   solution 1: train a mini-batch
    def fitness(self):

    # encode and return evolution knowledge
    def teach(self):
        return 
        
    # decode and absorb evolution knowledge
    def learn(self, ):


class Generation:
    def __init__(self,
                population_size=10):
        # Individuals
        self._population = list()
        for i in range(population_size):
            self._population.append(Individual())

    @classmethod
    def select(self):

    def generate(self):

    # select and generate
    def evolve(self):
        selected_individuals = self.select()
        generated_individuals = self.generate()
        self._population = selected_individuals + generated_individuals

    def 

if __name__ == '__main__':
    if not os.path.exists(FLAGS.evolve_dir):
        os.mkdir(FLAGS.evolve_dir)
        print('Created evolution history directory', FLAGS.evolve_dir)
    generation = Generation()
    for epoch in range(FLAGS.epoch):
        generation.evolve()
