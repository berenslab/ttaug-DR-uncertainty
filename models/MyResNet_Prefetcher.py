import numpy as np
import tensorflow as tf

# from models.AbstractModel import AbstractModel
from utils.TensorUtil import resnet_bottleneck_head_block
from utils.TensorUtil import resnet_bottleneck_identity_block
from utils.TensorUtil import conv_prelu
from utils.TensorUtil import max_pool
from utils.TensorUtil import avg_pool
from utils.TensorUtil import fc_prelu
from utils.DataAugmentation import do_nothing, data_augmentation, data_augmentation_wrapper
from utils.Reader import AdvancedReader

from sklearn.metrics import roc_curve, auc


class MyResNetPrefetcher:

    type = 'ResNet_Bottleneck'

    def __init__(self, network_config, tr_reader=None, val_reader=None, model_path=None, resurrection_path=None,
                 name='ResNetPrefetcher'):
        self.network_config = network_config

        self.name = name
        self.resStackDescription = ''
        self.batch_shape = None
        self.num_weight_layers = 0
        self.num_trainable_params = None
        self.diagnostics = {}
        self.session = None  # For QueueRunner, this must be a Monitored Session
        # DISABLE the object's own graph! USE THE SAME DEFAULT GRAPH FOR ALL INSTANCE of the same config.
        # self.graph = tf.Graph().as_default()  # Default computation graph maintained by the network itself.

        # Some nodes of interest
        self.inputs = None  # possible placeholder
        self.labels = None  # possible placeholder
        self.labels_1hot = None
        self.logits = None
        self.penultimate_features = None
        self.predictions = None
        self.predictions_1hot = None
        self.loss = None
        self.learning_rate = None
        self.train_op = None
        self.saver = None
        self.resurrector = None
        self.init = None
        self.is_training = None  # possible placeholder
        self.momentum = None  # possible placeholder
        self.rmin = None  # possible placeholder
        self.rmax = None  # possible placeholder
        self.dmax = None  # possible placeholder

        # Extra nodes for test-time data augmentation. Not the best way, but currently it is what it is!
        self.ttaug_input = None
        self.ttaug_input_aug = None

        self.descriptor = None
        self.model_path = model_path  # Best model is saved at this location
        self.resurrection_path = resurrection_path  # Where the model is saved and reloaded from during training.
        # Resurrection avoids the slow-down of Tensorflow due to thread-caching, fragmentation, etc...

        # Readers for training and inference
        self.reader_tr = tr_reader
        self.reader_val = val_reader

        # Dataset from generator configuration
        self.iterator = None  # A common iterator, maybe a feedable one
        self.dataset_tr = None  # Then, respective datasets and iterators
        self.initializer_tr = None # Not needed if Feedable iterators used
        self.dataset_val = None
        self.initializer_val = None
        self.dataset_ttaug = None
        self.initializer_ttaug = None
        self.next_element = None

        self._current_iter_tr = None  # useful for momentum, rmax, rmin, dmax, etc..
        self._sampling = None
        # These are useful for generators
        self.output_dtypes = (tf.float32, tf.int32, tf.bool, tf.float32, tf.float32, tf.float32, tf.float32)
        self.output_shapes = None

        assert len(self.network_config['conv_depths']) == len(self.network_config['num_filters']), \
            'Convolutional stack depths do not match the number of filters per stack'

        self.num_weight_layers = self.num_weight_layers + 1  # conv1: 1 weight layer
        for i in range(len(self.network_config['conv_depths'])):
            self.num_weight_layers = self. num_weight_layers + self.network_config['conv_depths'][i] * \
                                     len(self.network_config['num_filters'][i])
            self.resStackDescription = self.resStackDescription + str(self.network_config['conv_depths'][i])
        self.num_weight_layers = self.num_weight_layers + len(self.network_config['fc_depths'])  # FC layers
        self.num_weight_layers = self.num_weight_layers + 1  # logits to softmax: 1 weight layer

        print('Total number of weight layers : %g' % self.num_weight_layers)
        print('Residual stack depths : %s ' % self.resStackDescription)

        self.descriptor = '_' + str(self.resStackDescription) + \
                          '_regConst_' + str(self.network_config['lambda']) + \
                          '_lr_' + str(self.network_config['lr']) + \
                          '_m_' + str(self.network_config['batch_size'])

        if self.network_config['data_aug']:
            self.descriptor = self.descriptor + '_dataAug_' + str(self.network_config['data_aug_prob'])

        self.descriptor = self.name + str(self.num_weight_layers) + self.descriptor

        if self.model_path is None:
            self.model_path = '../modelstore/' + self.descriptor  # + '.ckpt'
        else:
            self.model_path = self.model_path + self.descriptor  # + '.ckpt'
        if self.resurrection_path is None:
            self.resurrection_path = '../resurrection/' + self.descriptor  # + '_RESURRECTION'  # .ckpt'
        else:
            self.resurrection_path = self.resurrection_path + self.descriptor  # + '_RESURRECTION'  # .ckpt'

    # Since the generator is attached to a dataset with its arguments, dynamic input, such as iter, etc....
    def generator_train(self, batch_size, normalize, max_iter):
        oversampling_threshold = int(max_iter * self.network_config['oversampling_limit'])

        for _ in range(max_iter):
            if self._current_iter_tr < oversampling_threshold:
                self._sampling = 'balanced'
            else:
                self._sampling = 'stratified'

            x_batch, y_batch, _ = self.reader_tr.next_batch(batch_size=batch_size, normalize=normalize,
                                                            shuffle=True, sampling=self._sampling)

            progress = float(self._current_iter_tr) / float(max_iter)
            # momentum settings
            momentum_max = self.network_config['momentum_max']
            if progress <= 0.95:
                momentum = 1 - np.power(2, -1 - np.log2(np.floor(self._current_iter_tr / 250.) + 1))
            else:
                momentum = 0.5
            momentum = np.minimum(momentum, momentum_max)

            # Params for Batch Renormalization
            if progress < 0.01:  # up to this point, use BatchNorm alone
                rmax = 1.
                rmin = 1.
                dmax = 0.
            else:  # then, gradually increase the clipping values
                rmax = np.exp(1.5 * progress)  # 2.
                rmin = 1. / rmax
                dmax = np.exp(2.0 * progress) - 1  # 2.5
            if progress > 0.95:
                rmin = 0.

            yield (x_batch, y_batch, True, momentum, rmin, rmax, dmax)

    def generator_val(self, batch_size, normalize, quick_eval=False):
        i = 0
        while not self.reader_val.exhausted_test_cases:
            x_batch, y_batch, _ = self.reader_val.next_batch(batch_size=batch_size, normalize=normalize, shuffle=False)
            if quick_eval and i == 50:
                self.reader_val.exhausted_test_cases = True
            i += 1

            yield (x_batch, y_batch, False, 0, 1, 1, 0)

    def generator_ttaug(self, normalize, quick_eval=False):
        i = 0
        while not self.reader_val.exhausted_test_cases:
            org_ex, label, _ = self.reader_val.next_batch(batch_size=1, normalize=normalize, shuffle=False)
            feed_img = {self.ttaug_input: org_ex}
            images = []
            labels = []
            for k in range(self.network_config['T']):
                aug_ex = np.squeeze(self.session.run([self.ttaug_input_aug], feed_dict=feed_img))
                images.append(aug_ex)
                labels.append(label)

            x_batch = np.reshape(np.asarray(images, dtype=np.float32), [-1, 512, 512, 3])
            y_batch = np.reshape(np.asarray(labels, dtype=np.float32), [-1, 1])

            if quick_eval and i == 50:
                self.reader_val.exhausted_test_cases = True
            i += 1

            yield (x_batch, y_batch, False, 0, 1, 1, 0)

    def build(self, expose_inputs=False):

        print('Building the model graph...')

        conv_stack_depths = self.network_config['conv_depths']
        num_filters = self.network_config['num_filters']
        fc_depths = self.network_config['fc_depths']
        lambda_ = self.network_config['lambda']
        learning_rate = self.network_config['lr']
        decay_steps = self.network_config['decay_steps']
        decay_rate = self.network_config['decay_rate']
        data_aug = self.network_config['data_aug']
        data_aug_prob = self.network_config['data_aug_prob']

        # Now, construct the ResNet architecture
        print('Instance shape : ' + str(self.network_config['instance_shape']))
        self.batch_shape = [None]
        for num in self.network_config['instance_shape']:
            self.batch_shape.append(num)
        print('Batch shape : ' + str(self.batch_shape))
        print('Num. of classes : ' + str(self.network_config['num_classes']))
        self.output_shapes = (self.batch_shape, [None, 1], [], [], [], [], [])

        # First, some control mechanism to drive the network
        with tf.name_scope('cockpit'):
            # For faster training/validation, isolate the inputs (placeholders) and use the generators
            if not expose_inputs:
                # Use CPU for iterators and prefetching. Hmmm, leads to slow performance with less GPU usage???
                # with tf.device('/cpu:0'):
                # TRAINING DATA GENERATOR
                self.dataset_tr = tf.data.Dataset.from_generator(generator=self.generator_train,
                                                                 output_types=self.output_dtypes,
                                                                 output_shapes=self.output_shapes,
                                                                 args=([self.network_config['batch_size'],
                                                                        True, self.network_config['max_iter']]))
                self.dataset_tr = self.dataset_tr.prefetch(buffer_size=self.network_config['dataset_buffer_size'])
                # batch(batch_size=1)

                # VALIDATION DATA GENERATOR, used while training as well as inference
                self.dataset_val = tf.data.Dataset.from_generator(generator=self.generator_val,
                                                                  output_types=self.output_dtypes,
                                                                  output_shapes=self.output_shapes,
                                                                  args=([self.network_config['batch_size'],
                                                                         True, self.network_config['quick_dirty_val']]))
                self.dataset_val = self.dataset_val.prefetch(buffer_size=self.network_config['dataset_buffer_size'])
                #  batch(batch_size=1)

                # TTAUG DATA GENERATOR
                self.dataset_ttaug = tf.data.Dataset.from_generator(generator=self.generator_ttaug,
                                                                    output_types=self.output_dtypes,
                                                                    output_shapes=self.output_shapes,
                                                                    args=(True, self.network_config['quick_dirty_val']))
                self.dataset_ttaug = self.dataset_ttaug.prefetch(buffer_size=self.network_config['dataset_buffer_size'])
                # batch(batch_size=1)

                # Common iterator for both datasets
                # Iterator has to have same output types across all Datasets to be used
                self.iterator = tf.data.Iterator.from_structure(self.dataset_tr.output_types,
                                                                self.dataset_tr.output_shapes)
                # Initialize with required Datasets NOT NEEDED if feedable iterator is used!
                self.initializer_tr = self.iterator.make_initializer(self.dataset_tr, name='initializer_tr')
                self.initializer_val = self.iterator.make_initializer(self.dataset_val, name='initializer_val')
                self.initializer_ttaug = self.iterator.make_initializer(self.dataset_ttaug, name='initializer_ttaug')

                # Saving of STATEFUL FUNCTIONs is not supported yet.... SO, DO NOT SAVE the iterator state...
                # saveable = tf.contrib.data.make_saveable_from_iterator(self.iterator)
                # tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)

            # For deployment, expose the inputs (placeholders) and resort to the feed_dict mechanism
            if expose_inputs:
                self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')
                self.momentum = tf.placeholder(dtype=tf.float32, shape=[], name='momentum_coefficient')
                # self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob_for_dropout')

        with tf.name_scope('layer0'):
            if not expose_inputs:
                [self.inputs, self.labels, self.is_training, self.momentum, self.rmin, self.rmax, self.dmax] = \
                    self.iterator.get_next(name="next_element")
            else:
                self.inputs = tf.placeholder(dtype=tf.float32, shape=self.batch_shape, name='inputs')
                self.labels = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='labels')
                self.rmin = tf.placeholder(dtype=tf.float32, shape=[1], name='renorm_clip_rmin')
                self.rmax = tf.placeholder(dtype=tf.float32, shape=[1], name='renorm_clip_rmax')
                self.dmax = tf.placeholder(dtype=tf.float32, shape=[1], name='renorm_clip_dmax')

            self.labels_1hot = tf.squeeze(tf.one_hot(indices=self.labels, depth=self.network_config['num_classes'],
                                                     name='labels_1hot'))

            # branching for data augmentation operations in training
            if data_aug:
                inputs_aug = tf.cond(self.is_training,
                                     true_fn=lambda: data_augmentation_wrapper(self.inputs,
                                                                               data_aug_prob=data_aug_prob),
                                     false_fn=lambda: do_nothing(self.inputs), name='data_augmentation_if_training'
                                     )
                inputs2next = inputs_aug
            else:
                inputs2next = self.inputs

            # What if the standardization reduces the impact of data augmentation? Overthinking! Just use it!!!!
            # For instance, random contrast increases/decreases the contrast between pixels, but standardization
            # eliminates such changes.
            inputs2next = tf.map_fn(lambda img: tf.image.per_image_standardization(img), inputs2next,
                                    dtype=tf.float32,
                                    name='image_standardization')

        # Placeholder to be filled at test-time and by an outsider!
        scope = 'ttaug'
        with tf.name_scope(scope):
            self.ttaug_input = tf.placeholder(dtype=tf.float32, shape=self.batch_shape, name='ttaug_input')
            self.ttaug_input_aug = data_augmentation(self.ttaug_input)

        scope = 'conv1'  # the very first 5x5 convolution and pooling operations on inputs
        with tf.variable_scope(scope):
            print(scope)
            inputs2next = conv_prelu(inputs=inputs2next,
                                     kernel_shape=[7, 7], num_filters=num_filters[0][0], strides=[2, 2],
                                     reg_const=lambda_, is_training=self.is_training,
                                     rmin=self.rmin, rmax=self.rmax, dmax=self.dmax
                                     )
            inputs2next = max_pool(inputs2next, kernel_shape=[3, 3], strides=[2, 2], padding='SAME')

        # Iterate over convolutional stacks
        for stack_idx in range(len(conv_stack_depths)):
            scope = 'conv' + str(stack_idx+2)
            with tf.variable_scope(scope):  # e.g., conv2
                head_placed = False
                # Iterate over residual block in each conv. stack
                for i in range(conv_stack_depths[stack_idx]):
                    with tf.variable_scope(str(i + 1)):  # e.g., conv2/1
                        print(scope + '/' + str(i + 1))
                        if not head_placed:  # Place a head block first.
                            if stack_idx == 0:
                                head_conv_stride = [1, 1]
                            else:
                                head_conv_stride = [2, 2]
                            inputs2next = resnet_bottleneck_head_block(inputs2next,
                                                                       kernel_shapes=[[1, 1], [3, 3], [1, 1]],
                                                                       num_filters=num_filters[stack_idx],
                                                                       strides=head_conv_stride, padding='SAME',
                                                                       reg_const=lambda_, is_training=self.is_training,
                                                                       rmin=self.rmin, rmax=self.rmax, dmax=self.dmax
                                                                       )
                            head_placed = True
                        else:
                            inputs2next = resnet_bottleneck_identity_block(inputs2next,
                                                                           kernel_shapes=[[1, 1], [3, 3], [1, 1]],
                                                                           num_filters=num_filters[stack_idx],
                                                                           strides=[1, 1], padding='SAME',
                                                                           reg_const=lambda_,
                                                                           is_training=self.is_training,
                                                                           rmin=self.rmin, rmax=self.rmax,
                                                                           dmax=self.dmax
                                                                           )

        # concatenate features from global max pooling and global avg. pooling
        # and flatten them before the FC layers, or logits if no FC layers is present.
        inputs2next = tf.concat([tf.layers.flatten(max_pool(inputs2next, kernel_shape=[16, 16],
                                                            strides=[1, 1], padding='VALID'),
                                                   name='flattened_max_pool_feat'),
                                 tf.layers.flatten(avg_pool(inputs2next, kernel_shape=[16, 16],
                                                            strides=[1, 1], padding='VALID'),
                                                   name='flattened_avg_pool_feat')
                                 ],
                                axis=-1,
                                name='concat_glob_max_glob_avg')

        fan_in = inputs2next.get_shape().as_list()[-1]  # num_filters[-1][-1]
        for fc_idx in range(len(fc_depths)):
            scope = 'fc' + str(fc_idx+1)
            print(scope)
            if fc_idx == (len(fc_depths)-1):  # sparsity in the last layer before softmax
                reg_type = 'l1'
            else:
                reg_type = 'l2'
            with tf.variable_scope(scope):
                fan_out = fc_depths[fc_idx]
                inputs2next = fc_prelu(inputs=inputs2next, fan_in=fan_in, fan_out=fan_out,
                                       reg_const=lambda_, reg_type=reg_type, is_training=self.is_training,
                                       rmin=self.rmin, rmax=self.rmax, dmax=self.dmax
                                       )
                fan_in = fan_out

        # Now, the final layer outputs logits
        scope = 'logits'
        print(scope)
        with tf.variable_scope(scope):
            self.penultimate_features = inputs2next  # the penultimate layer's activations

            print('\t[%d, %d]' % (inputs2next.get_shape().as_list()[-1], self.network_config['num_classes']))

            # Variables #
            weights = tf.get_variable(name='weights', shape=[inputs2next.get_shape().as_list()[-1],
                                                             self.network_config['num_classes']],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                                  stddev=tf.sqrt(2.0 / inputs2next.get_shape().as_list()[-1])
                                                                                  ),
                                      regularizer=tf.contrib.layers.l2_regularizer(scale=lambda_)
                                      )

            biases = tf.get_variable(name='biases', shape=[self.network_config['num_classes']], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0),
                                     regularizer=None
                                     )

            # Operations #
            self.logits = tf.add(tf.matmul(inputs2next, weights), biases, name='logits')

            # This node may be changed to tf.nn.sigmoid or softmax depending on the Xentropy used for training.
            self.predictions_1hot = tf.nn.softmax(self.logits, name='predictions_1hot')
            # self.predictions = tf.argmax(self.predictions_1hot, axis=1, name='predictions')

        scope = 'loss'
        with tf.name_scope(scope):
            Xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(self.labels_1hot,
                                                                                          name='labels_1hot_stopgrad'),
                                                                  logits=self.logits,
                                                                  name='Xentropy')
            #Xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_1hot,
            #                                                   logits=self.logits,
            #                                                   name='Xentropy')

            # epsilon = tf.constant(value=1e-14, name='epsilon')
            # Xentropy = self.labels_1hot * (-tf.log(self.predictions_1hot + epsilon)) * self.pos_weights

            # sample_weights = tf.reduce_sum(tf.multiply(self.labels_1hot, self.pos_weights), 1)
            # Xentropy = tf.losses.softmax_cross_entropy(onehot_labels=self.labels_1hot, logits=self.logits,
            # weights=sample_weights)

            self.loss = tf.reduce_mean(Xentropy, name='mean_Xentropy')

        scope = 'train_op'
        with tf.name_scope(scope):
            global_step = tf.Variable(0, trainable=False)
            # self.learning_rate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step,
            #                                                 decay_steps=decay_steps, decay_rate=decay_rate,
            #                                                 staircase=True, name='exp_lr_decay'
            #                                                 )
            # self.learning_rate = tf.train.cosine_decay(learning_rate=learning_rate, global_step=global_step,
            #                                            decay_steps=decay_steps, alpha=0.1, name='cos_lr_decay'
            #                                            )
            self.learning_rate = tf.train.cosine_decay_restarts(learning_rate=learning_rate, global_step=global_step,
                                                                first_decay_steps=decay_steps,
                                                                t_mul=2.0, m_mul=decay_rate,
                                                                alpha=0.01, name='cos_lr_decay_warmrestart'
                                                                )

            # self.learning_rate = tf.train.linear_cosine_decay(learning_rate=learning_rate, global_step=global_step,
            #                                                   decay_steps=decay_steps,
            #                                                   num_periods=0.5, alpha=0.0, beta=0.001,
            #                                                   name='lin_cos_lr_decay'
            #                                                   )

            # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='adam_optimizer')
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum,
                                                   use_nesterov=True,
                                                   name='nesterov_momentum_optimizer')

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.minimize(self.loss, global_step=global_step)

        scope = 'saver'
        with tf.name_scope(scope):
            self.saver = tf.train.Saver()
            ## NO FILE PRESENT during the first build --> ERROR
            # self.resurrector = tf.train.import_meta_graph(self.resurrection_path + '.meta')

        scope = 'init'
        with tf.name_scope(scope):
            self.init = tf.global_variables_initializer()

    # End of build method #

    def initialize(self):
        self.session = tf.Session()
        self.session.run(self.init)

    def restore(self, expose_inputs=False):
        # assert self.graph is tf.get_default_graph(), "Network's graph is not the DEFAULT graph!"

        print('Restoring model from file : %s' % self.model_path)

        tf.reset_default_graph()  # explicitly clean up before the new graph!

        if self.session is None:  # e.g., not initialized in the case of RESTORE.
            self.session = tf.Session()

        self.build(expose_inputs=expose_inputs)
        self.saver.restore(self.session, self.model_path)

        if expose_inputs:
            print("Model restored and inputs are exposed.")
        else:
            print("Model restored without exposing inputs.")

    def train(self, train_source='/gpfs01/berens/user/mayhan/kaggle_dr_data/train_JF_BG_512/',
              train_csv_file='/gpfs01/berens/user/mayhan/kaggle_dr_data/trainLabels.csv',
              val_source='/gpfs01/berens/user/mayhan/kaggle_dr_data/test_JF_BG_512/',
              solution_csv_file='/gpfs01/berens/user/mayhan/kaggle_dr_data/retinopathy_solution.csv'):

        max_iter = self.network_config['max_iter']
        val_step = self.network_config['val_step']
        resurrection_step = self.network_config['resurrection_step']

        best_roc = 0.

        if max_iter <= 1000:
            step = 1
        else:
            step = int(max_iter / 1000)
        self.diagnostics['losses'] = []
        self.diagnostics['avg_losses'] = []
        self.diagnostics['val_roc1'] = []
        self.diagnostics['val_roc2'] = []
        self.diagnostics['multi_acc'] = []

        total_loss = 0.

        print('Training %s ...' % self.name)
        self.num_trainable_params = np.sum([np.product([xi.value for xi in x.get_shape()])
                                            for x in tf.trainable_variables()])
        print('Number of trainable parameters : %d' % self.num_trainable_params)

        self.reader_tr = AdvancedReader(source=train_source, file_type='.jpeg',
                                        csv_file=train_csv_file, mode='train')

        # self.session.run(self.initializer_tr)  # AdvancedReader object's internal state is independent of iterator.
        # So, the iterator can be reinitialized without disturbing the sampling of training minibatches.
        # Move it into the loop. Otherwise, val_init break tr_init within the loop!

        self.session.run(self.initializer_tr)  # initialize now and also after each validation

        for self._current_iter_tr in range(max_iter):
            # self.session.run(self.initializer_tr)  # Having it here might also be a good idea for RESURRECTION since
            # stateful functions are not saved by API...

            try:
                _, loss_value, lr, momentum = self.session.run([self.train_op, self.loss,
                                                                self.learning_rate, self.momentum])
                total_loss = total_loss + loss_value
                avg_loss = total_loss / (self._current_iter_tr + 1)
                self.diagnostics['losses'].append(loss_value)
                self.diagnostics['avg_losses'].append(avg_loss)
            except tf.errors.OutOfRangeError:
                print("End of training dataset")

            if self._current_iter_tr % step == 0:
                print("Iter %d/%d  Avg. Loss: %f  Cur. Loss: %f  L.R. : %g Mom.: %g %s" %
                      (self._current_iter_tr, max_iter, avg_loss, loss_value, lr, momentum, self._sampling))

            if self._current_iter_tr % val_step == 0:
                print("Iter %d/%d, Validation once in %d steps..." % (self._current_iter_tr, max_iter, val_step))
                _, _, roc_auc_val_onset1, roc_auc_val_onset2, multi_acc, _, _ = self.inference(source=val_source,
                                                                                               csv_file=solution_csv_file)
                self.session.run(self.initializer_tr)

                self.diagnostics['val_roc1'].append(roc_auc_val_onset1)
                self.diagnostics['val_roc2'].append(roc_auc_val_onset2)
                self.diagnostics['multi_acc'].append(multi_acc)

                if self.saver is not None and best_roc < roc_auc_val_onset1:
                    print('Current best : %g\t New best : %g' % (best_roc, roc_auc_val_onset1))
                    save_path = self.saver.save(self.session, self.model_path)
                    print("A better model found. Saving the model in path: %s" % save_path)
                    best_roc = roc_auc_val_onset1

            # save and reload the model graph
            if self._current_iter_tr != 0 and self._current_iter_tr % resurrection_step == 0 and self.saver is not None:
                print('Resurrection is due...')
                save_path = self.saver.save(self.session, self.resurrection_path)
                print("Saved the model in path: %s" % save_path)
                print('Reseting the default graph...')
                tf.reset_default_graph()  # explicitly clean up before the new graph!
                resurrector = tf.train.import_meta_graph(self.resurrection_path + '.meta')
                resurrector.restore(self.session, self.resurrection_path)
                resurrector = None  # Explicitly set to None so that some memory may be freed over long iterations

        _, _, roc_auc_val_onset1, roc_auc_val_onset2, multi_acc, _, _ = self.inference(source=val_source,
                                                                                       csv_file=solution_csv_file)
        # self.session.run(self.initializer_tr)  # end of training. No need to initialize training iterator

        self.diagnostics['val_roc1'].append(roc_auc_val_onset1)
        self.diagnostics['val_roc2'].append(roc_auc_val_onset2)
        self.diagnostics['multi_acc'].append(multi_acc)

        if self.saver is not None and best_roc < roc_auc_val_onset1:
            print('Current best : %g\t New best : %g' % (best_roc, roc_auc_val_onset1))
            save_path = self.saver.save(self.session, self.model_path)
            print("Saving the model in path: %s" % save_path)
            best_roc = roc_auc_val_onset1

        print('Average batch loss after %d iterations : %f' % (max_iter, avg_loss))
        print('Last batch loss after %d iterations : %f' % (max_iter, loss_value))

    def inference(self, source='/gpfs01/berens/user/mayhan/kaggle_dr_data/test_JF_BG_512/',
                  csv_file='/gpfs01/berens/user/mayhan/kaggle_dr_data/retinopathy_solution.csv',
                  mode='val'):
        labels_1hot = []
        predictions_1hot = []
        penultimate_features = []
        logits_all = []

        self.reader_val = AdvancedReader(source=source, csv_file=csv_file, mode=mode)

        print('Evaluating %s with %g weight layers...' % (self.name, self.num_weight_layers))

        self.session.run(self.initializer_val)

        while True:  # Until the validation dataset is exhausted
            try:
                predictions, labels, features, logits = self.session.run([self.predictions_1hot, self.labels_1hot,
                                                                          self.penultimate_features, self.logits])
                labels_1hot.append(labels)
                predictions_1hot.append(predictions)
                penultimate_features.append(features)
                logits_all.append(logits)
            except tf.errors.OutOfRangeError:
                break

        logits_all = np.squeeze([item for sublogits_all in logits_all for item in sublogits_all])
        penultimate_features = np.squeeze([item for subpenultimate_features in penultimate_features for item in subpenultimate_features])
        labels_1hot = np.squeeze([item for sublabels_1hot in labels_1hot for item in sublabels_1hot])
        predictions_1hot = np.squeeze([item for subpredictions_1hot in predictions_1hot for item in subpredictions_1hot])
        correct = np.equal(np.argmax(labels_1hot, axis=1), np.argmax(predictions_1hot, axis=1))
        acc = np.mean(np.asarray(correct, dtype=np.float32))
        print('Multi-class Accuracy : %.5f' % acc)

        onset_level = 1
        labels_bin = np.greater_equal(np.argmax(labels_1hot, axis=1), onset_level)
        pred = np.sum(predictions_1hot[:, onset_level:], axis=1)
        fpr, tpr, _ = roc_curve(labels_bin, pred)
        roc_auc_onset1 = auc(fpr, tpr)
        print('Onset level = %d\t ROC-AUC: %.5f' % (onset_level, roc_auc_onset1))

        onset_level = 2
        labels_bin = np.greater_equal(np.argmax(labels_1hot, axis=1), onset_level)
        pred = np.sum(predictions_1hot[:, onset_level:], axis=1)
        fpr, tpr, _ = roc_curve(labels_bin, pred)
        roc_auc_onset2 = auc(fpr, tpr)
        print('Onset level = %d\t ROC-AUC: %.5f' % (onset_level, roc_auc_onset2))

        return labels_1hot, predictions_1hot, roc_auc_onset1, roc_auc_onset2, acc, penultimate_features, logits_all

    def inference_ttaug(self, source='/gpfs01/berens/user/mayhan/kaggle_dr_data/test_JF_BG_512/',
                        csv_file='/gpfs01/berens/user/mayhan/kaggle_dr_data/retinopathy_solution.csv',
                        mode='val'):
        labels_all = []
        predictions_all = []
        features_all = []
        logits_all = []

        self.reader_val = AdvancedReader(source=source, csv_file=csv_file, mode=mode)

        print('Evaluating %s with %g weight layers with Test-Time Data Augmentation...'
              % (self.name, self.num_weight_layers))

        self.session.run(self.initializer_ttaug)

        while True:  # Until the validation dataset is exhausted
            try:
                predictions, labels, features, logits = self.session.run([self.predictions_1hot, self.labels_1hot,
                                                                          self.penultimate_features, self.logits])
                labels_all.append(labels[0])
                predictions_all.append(predictions)
                features_all.append(features)
                logits_all.append(logits)
            except tf.errors.OutOfRangeError:
                break

        # Convert from a list of M items of size Tx5 to an array of dims MxTx5. For labels_1hot: Mx5.
        labels_1hot = np.asarray(labels_all)
        predictions_all = np.asarray(predictions_all)
        features_all = np.asarray(features_all)  # MxTx1024
        logits_all = np.asarray(logits_all)  # MxTx5

        # use the median of T predictions for the final class membership: Mx1x5
        predictions_1hot_median = np.median(predictions_all, axis=1)
        correct = np.equal(np.argmax(labels_1hot, axis=1), np.argmax(predictions_1hot_median, axis=1))
        acc = np.mean(np.asarray(correct, dtype=np.float32))
        print('Median Accuracy (multi-class) : %.5f' % acc)

        predictions_1hot_mean = np.mean(predictions_all, axis=1)
        correct = np.equal(np.argmax(labels_1hot, axis=1), np.argmax(predictions_1hot_mean, axis=1))
        acc = np.mean(np.asarray(correct, dtype=np.float32))
        print('Mean Accuracy (multi-class) : %.5f' % acc)

        onset_level = 1
        print('Onset level = %d' % onset_level)
        labels_bin = np.greater_equal(np.argmax(labels_1hot, axis=1), onset_level)
        pred_bin = np.sum(predictions_all[:, :, onset_level:], axis=2)  # MxTx1
        pred_bin_median = np.median(pred_bin, axis=1)  # Mx1x1
        fpr, tpr, _ = roc_curve(labels_bin, np.squeeze(pred_bin_median))
        roc_auc_onset1_median = auc(fpr, tpr)
        print('With median pred., ROC-AUC: %.5f' % roc_auc_onset1_median)
        pred_bin_mean = np.mean(pred_bin, axis=1)  # Mx1x1
        fpr, tpr, _ = roc_curve(labels_bin, np.squeeze(pred_bin_mean))
        roc_auc_onset1_mean = auc(fpr, tpr)
        print('With mean pred., ROC-AUC: %.5f' % roc_auc_onset1_mean)

        onset_level = 2
        print('Onset level = %d' % onset_level)
        labels_bin = np.greater_equal(np.argmax(labels_1hot, axis=1), onset_level)
        pred_bin = np.sum(predictions_all[:, :, onset_level:], axis=2)  # MxTx1
        pred_bin_median = np.median(pred_bin, axis=1)  # Mx1x1
        fpr, tpr, _ = roc_curve(labels_bin, np.squeeze(pred_bin_median))
        roc_auc_onset2_median = auc(fpr, tpr)
        print('With median pred., ROC-AUC: %.5f' % roc_auc_onset2_median)
        pred_bin_mean = np.mean(pred_bin, axis=1)  # Mx1x1
        fpr, tpr, _ = roc_curve(labels_bin, np.squeeze(pred_bin_mean))
        roc_auc_onset2_mean = auc(fpr, tpr)
        print('With mean pred., ROC-AUC: %.5f' % roc_auc_onset2_mean)

        return labels_1hot, predictions_all, features_all, logits_all

    def finalize(self):
        # Model saving features may be added here.
        #if self.saver is not None:
        #    # Save the variables to disk.
        #    save_path = self.saver.save(self.session, self.model_path)
        #    print("Model saved in path: %s" % save_path)

        self.session.close()
        self.session = None
        tf.reset_default_graph()
        # self.graph = None
