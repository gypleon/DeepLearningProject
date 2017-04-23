from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import copy
import numpy as np
import tensorflow as tf

import model
from data_reader import load_data, load_mini_data, DataReader


flags = tf.flags

# system
flags.DEFINE_integer('num_gpus', 1, 'the number of GPUs in the system')

# data
flags.DEFINE_integer('num_partitions',  100,  'total number of partitions for fitness training')
flags.DEFINE_string ('data_dir',        'data',   'data directory. Should contain train.txt/valid.txt/test.txt with input data')
flags.DEFINE_string ('population_dir',  'population', 'evolution history, information for generations')

# model params
flags.DEFINE_integer('char_embed_size', 15,                             'dimensionality of character embeddings')
flags.DEFINE_float  ('dropout',         0.5,                            'dropout. 0 = no dropout')
flags.DEFINE_integer('highway_layers',  2,                              'number of highway layers')

# evolution configuration
flags.DEFINE_integer('num_winners',             3, 'number of winners of each generation')
flags.DEFINE_integer('population_size',         10, 'number of individuals of each generation')
flags.DEFINE_integer('max_evo_epochs',          15, 'max number of evolution iterations')
flags.DEFINE_float  ('learning_threshold',      0.001, 'similarity threshold for teacher selection')
flags.DEFINE_float  ('prob_mutation_struct',    0.1, 'probability of mutation for individual structures')
flags.DEFINE_float  ('prob_mutation_param',     0.1, 'probability of mutation for individual parameters')
flags.DEFINE_integer('max_cnn_filter_types',    30, 'max number of cnn filter types')
flags.DEFINE_integer('max_cnn_type_filters',    300, 'max number of cnn filters for a specific type')
flags.DEFINE_integer('max_rnn_layers',          3, 'max number of rnn layers')
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


class adict(dict):
    def __init__(self):
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

        self._individual_dir = FLAGS.population_dir + "/individual_%d" % self._id_number
        if not os.path.exists(self._individual_dir):
            os.mkdir(self._individual_dir)

        # TODO: generate individual seed
        self._seed = np.random.seed(id_number * 13)
        self._id_number = id_number
        self._max_word_length = 0
        self._fitness = 0

        # layer_i:          { filter_type_1, ..., filter_type_n } 
        # filter_type_j:    [ size, number ]
        # size, number: integer
        self._cnn_layer = cnn_layer

        # _rnn_layers:  { layer_1, ..., layer_n }
        # layer_i:      [ size ]
        self._rnn_layers = rnn_layers

        # encode network structure
        self._knowledge = adict(
                        char_embed_size = char_embed_size,
                        dropout = dropout,
                        structure = self.encode_structure()
                        )
        self._struct_exp = self.experience(self._knowledge.structure)

        # create model
        self._gpu_id = self._id_number % FLAGS.num_gpus
        self._graph = tf.Graph()
        # self._model, self._valid_model, self._saver = self.create_graph()

    def __del__(self):
        self._graph.close()

    @property
    def get_exp(self):
        return self._struct_exp

    @property
    def get_fitness(self):
        return self._fitness

    @classmethod
    def encode_structure(self):
        struct_cnn = np.zeros([FLAGS.max_cnn_filter_types], dtype=np.int32)
        struct_rnn = np.zeros([FLAGS.max_rnn_layers], dtype=np.int32)
        # vector for CNN
        for filter_type in self._cnn_layer.values():
            struct_cnn[filter_type[0]-1] = filter_type[1]
        # vector for RNN
        for layer_i, num_units in self._rnn_layers.items():
            struct_rnn[int(layer_i)-1] = num_units
        self._knowledge.structure = [copy.deepcopy(struct_cnn), copy.deepcopy(struct_rnn)]
        return self._knowledge.structure

    def decode_structure(self, knowledge):
        cnn_layer = {}
        rnn_layers = {}
        struct_cnn = knowledge.structure[0]
        struct_rnn = knowledge.structure[1]
        for filter_type in range(struct_cnn.shape[0]):
            if struct_cnn[filter_type] > 0:
                cnn_layer['%d' % (filter_type+1)] = [filter_type+1, int(struct_cnn[filter_type])]
        for layer in range(struct_rnn.shape[0]):
            if struct_rnn[layer] > 0:
                rnn_layers['%d' % (layer+1)] = [int(struct_rnn[layer])]
        self._cnn_layer = cnn_layer
        self._rnn_layers = rnn_layers
        return self._cnn_layer, self._rnn_layers

    def experience(self, struct):
        self._struct_exp.append(struct) 
        return self._struct_exp

    def create_graph(self):
        with self._graph.as_default():
            with tf.device('/gpu:%d' % self._gpu_id):
                initializer = tf.random_uniform_initializer(-FLAGS.param_init, FLAGS.param_init)
                with tf.variable_scope("Individual_%d" % self._id_number, initializer=initializer):
                    my_model = model.individual_graph(
                                            char_vocab_size=self._char_vocab.size,
                                            word_vocab_size=self._word_vocab.size,
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

                saver = tf.train.Saver(max_to_keep=1)

                with tf.variable_scope("Individual_%d" % self._id_number, reuse=True):
                    valid_model = model.individual_graph(
                                            char_vocab_size=self._char_vocab.size,
                                            word_vocab_size=self._word_vocab.size,
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

    # TODO: currently just reconstruct a graph without reusing parameters
    def update_graph(self, word_vocab, char_vocab):
        with self._graph.as_default():
            with tf.device('/gpu:%d' % self._gpu_id):
                initializer = tf.random_uniform_initializer(-FLAGS.param_init, FLAGS.param_init)
                with tf.variable_scope("Individual_%d" % self._id_number, initializer=initializer):
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

                saver = tf.train.Saver(max_to_keep=1)

                with tf.variable_scope("Individual_%d" % self._id_number, reuse=True):
                    valid_model = model.individual_graph(
                                            char_vocab_size=char_vocab.size,
                                            word_vocab_size=word_vocab.size,
                                            char_embed_size=knowledge.char_embed_size,
                                            batch_size=FLAGS.batch_size,
                                            max_word_length=FLAGS.max_word_length,
                                            num_unroll_steps=FLAGS.num_unroll_steps,
                                            num_highway_layers=2,
                                            cnn_layer=self._cnn_layer,
                                            rnn_layers=self._rnn_layers,
                                            dropout=self._knowledge.dropout)
                    valid_model.update(model.loss_graph(valid_model.logits, FLAGS.batch_size, FLAGS.num_unroll_steps))
        return my_model, valid_model, saver

    # TODO: bias on less parameters
    def mutation_struct(self):

        # mutate cnn
        # mutate number of filters
        for filter_type in self._cnn_layer.values():
            num_var_cnn_type_filters = np.random.randint(-50, 51)
            # add / remove filters
            if num_var_cnn_type_filters > 0 or filter_type[1] + num_var_cnn_type_filters > 0:
                filter_type[1] += num_var_cnn_type_filters
        # mutate filter types
        num_var_cnn_filter_types = np.random.randint(-1, 2)
        # add filter type
        if num_var_cnn_filter_types > 0 and len(self._cnn_layer) + num_var_cnn_filter_types <= min(FLAGS.max_cnn_filter_types, self._max_word_length):
            available_types = list(set(filter_type for filter_type in range(1, FLAGS.max_cnn_filter_types+1))-set(filter_type[0] for filter_type in self._cnn_layer.values()))
            new_type = np.random.choice(available_types)
            num_new_type = np.random.randint(1, FLAGS.max_cnn_type_filters+1)
            self._cnn_layer[str(new_type)] = num_new_type
        # remove filter type
        elif num_var_cnn_filter_types < 0 and len(self._cnn_layer) + num_var_cnn_filter_types > 0:
            existed_types = [filter_type[0] for filter_type in self._cnn_layer.values()]
            selected_type = np.random.choice(existed_types)
            self._cnn_layer.pop('%d' % selected_type)
            
        # mutate rnn
        # mutate number of units
        num_var_rnn_layer_units = np.random.randint(-100, 101)
        for layer in self._rnn_layers.values():
            if layer[0] + num_var_rnn_layer_units > 0:
                layer[0] += num_var_rnn_layer_units
        # mutate number of rnn layers 
        num_var_rnn_layers = np.random.randint(-1, 2)
        # add rnn layer
        if num_var_rnn_layers > 0 and len(self._rnn_layers) + num_var_rnn_layers <= FLAGS.max_rnn_layers:
            for i in range(FLAGS.max_rnn_layers):
                if not self._rnn_layers.get('%d' % i+1):
                    num_units = np.random.randint(550, 750)
                    self._rnn_layers['%d' % i+1] = [num_units]
                    break
        # remove rnn layer
        elif num_var_rnn_layers < 0 and len(self._rnn_layers) + num_var_rnn_layers > 0:
            existed_layers = [layer for layer in self._rnn_layers.keys()]
            selected_layer = np.random.choice(existed_layers)
            self._rnn_layers.pop(selected_layer)

        # refresh knowledge
        self._knowledge.structure = self.encode_structure()
        self._struct_exp = self.experience(self._knowledge.structure)

    def mutation_param(self):
        # TODO: knowledge should be learned instead
        self._knowledge.char_embed_size = FLAGS.char_embed_size + np.random.randint(-FLAGS.char_embed_size, FLAGS.char_embed_size+1)
        self._knowledge.dropout = np.random.uniform()

    def mutation(self):
        # TODO: mutate parameters
        self.mutation_param()
        self.mutation_struct()

    # train on mini-dataset
    def fitness(self, partition, word_vocab, char_vocab, word_tensors, char_tensors, max_word_length):
        self._model, self._valid_model, self._saver = self.update_graph(word_vocab, char_vocab)

        # initialize model
        with tf.Session(graph=self._graph, config=tf.ConfigProto(log_device_placement=True)) as session:
            tf.global_variables_initializer().run()
            session.run(self._model.clear_char_embedding_padding)
            print('Created and initialized fresh individual_%d. Size: %d' % (self._id_number, self._model.model_size()))
            self._summary_writer = tf.summary.FileWriter(self._individual_dir, graph=session.graph)
            session.run(
                tf.assign(self._model.learning_rate, FLAGS.learning_rate),
            )
        
        self._max_word_length = max_word_length

        train_reader = DataReader(word_tensors['train'], char_tensors['train'], FLAGS.batch_size, FLAGS.num_unroll_steps)
        valid_reader = DataReader(word_tensors['valid'], char_tensors['valid'], FLAGS.batch_size, FLAGS.num_unroll_steps)

        with tf.Session(graph=self._graph, config=tf.ConfigProto(log_device_placement=True)) as session:
            best_valid_loss = None
            rnn_state = session.run(self._model.initial_rnn_state)
            for epoch in range(FLAGS.max_epochs):
                epoch_start_time = time.time()
                avg_train_loss = 0.0
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
                    avg_train_loss += 0.05 * (loss - avg_train_loss)
                    time_elapsed = time.time() - start_time
                    if count % FLAGS.print_every == 0:
                        print('%6d: %d [%5d/%5d], train_loss/perplexity = %6.8f/%6.7f secs/batch = %.4fs, grad.norm=%6.8f' % (step,
                                                                epoch, count,
                                                                train_reader.length,
                                                                loss, np.exp(loss),
                                                                time_elapsed,
                                                                gradient_norm))
                print('Epoch training time:', time.time()-epoch_start_time)
                # epoch done: time to evaluate
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
                    if count % FLAGS.print_every == 0:
                        print("\t> validation loss = %6.8f, perplexity = %6.8f" % (loss, np.exp(loss)))
                    avg_valid_loss += loss / valid_reader.length
                print("at the end of epoch:", epoch)
                print("train loss = %6.8f, perplexity = %6.8f" % (avg_train_loss, np.exp(avg_train_loss)))
                print("validation loss = %6.8f, perplexity = %6.8f" % (avg_valid_loss, np.exp(avg_valid_loss)))
                save_as = '%s/epoch%03d_%.4f.model' % (self._individual_dir, epoch, avg_valid_loss)
                self._saver.save(session, save_as)
                print('Saved model', save_as)
                ''' write out summary events '''
                summary = tf.Summary(value=[
                    tf.Summary.Value(tag="train_loss", simple_value=avg_train_loss),
                    tf.Summary.Value(tag="valid_loss", simple_value=avg_valid_loss)
                ])
                self._summary_writer.add_summary(summary, step)
                ''' decide if need to decay learning rate '''
                if best_valid_loss is not None and np.exp(avg_valid_loss) > np.exp(best_valid_loss) - FLAGS.decay_when:
                    print('validation perplexity did not improve enough, decay learning rate')
                    current_learning_rate = session.run(self._model.learning_rate)
                    print('learning rate was:', current_learning_rate)
                    current_learning_rate *= FLAGS.learning_rate_decay
                    if current_learning_rate < 1.e-5:
                        print('learning rate too small - stopping now')
                        break
                    session.run(train_model.learning_rate.assign(current_learning_rate))
                    print('new learning rate is:', current_learning_rate)
                else:
                    best_valid_loss = avg_valid_loss

        self._fitness = best_valid_loss
        return self._fitness

    def absorb(self, benefits):
        self._struct

    # encode and return evolution knowledge
    def teach(self):
        return self._knowledge
        
    # decode and absorb evolution knowledge
    def learn(self, knowledge):
        # self._knowledge = copy.deepcopy(knowledge)
        self._knowledge.char_embed_size = knowledge.char_embed_size
        self._knowledge.dropout = knowledge.dropout
        self._knowledge.structure = self.absorb(knowledge.structure)

    def save_model(self):
        self._saver
        return

    def train(self):
        return


class Population:
    def __init__(self,
                num_winners = 3,
                population_size = 10):

        self._num_winners = num_winners
        self._population_size = population_size

        # Individuals
        self._population = list()
        for i in range(self._population_size):
            self._population.append(self.generate(i))
        print("Initialized Population")

        self._average_fitness = None

    @property
    def average_fitness(self):
        self._average_fitness = 0 
        for individual in self._population:
            self._average_fitness += individual.get_fitness()
        self._average_fitness /= self._population_size
        return self._average_fitness

    def final_winner(self):
        return self.select()[0]

    @classmethod
    def generate(self, id_number):
        individual = Individual(id_number=id_number)
        print("Generated Individual_%d" % id_number)
        return individual

    def select(self):
        partition = np.random.randint(1, FLAGS.num_partitions+1)
        word_vocab, char_vocab, word_tensors, char_tensors, max_word_length = load_mini_data(FLAGS.data_dir, FLAGS.max_word_length, eos=FLAGS.EOS, partition=partition, num_partitions=FLAGS.num_partitions)
        for individual in self._population:
            print("Compute individual_%d's fitness: %f" % (i, individual.fitness(word_vocab, char_vocab, word_tensors, char_tensors, max_word_length)))
        self._population.sort(key=lambda individual:individual.get_fitness())
        winners = self._population[:self._num_winners]
        return winners

    def similarity(self, individual_1, individual_2):
        time_discount = 0.9
        dissim_cnn, dissim_rnn = 0, 0
        exp1 = individual_1.get_exp()
        exp2 = individual_2.get_exp()
        for struct1, struct2 in zip(exp1, exp2):
            dissim_cnn = time_discount * (dissim_cnn + np.linalg.norm(struct1[0] - struct2[0]))
            dissim_rnn = time_discount * (dissim_rnn + np.linalg.norm(struct1[1] - struct2[1]))
        print("Sim %d and %d is: %f" % (individual_1._id_number, individual_2._id_number, 1/(dissim_cnn+dissim_rnn)))
        return 1 / (dissim_cnn + dissim_rnn)

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
            # self._population[loser_id].update_graph()


def main(_):

    if not os.path.exists(FLAGS.population_dir):
        os.mkdir(FLAGS.population_dir)
        print('Created population history directory', FLAGS.population_dir)

    np.random.seed(seed=FLAGS.seed)

    # initialize population
    population = Population(num_winners=FLAGS.num_winners,
                            population_size=FLAGS.population_size)

    # perform evolution
    for epoch in range(FLAGS.max_evo_epochs):
        population.evolve()

    # get result
    result = population.final_winner()
    result.save_model()

    # TODO: fully train
    if FLAGS.if_train_winner == 1:
        result.train()


if __name__ == '__main__':
    tf.app.run()
