#Edited
#For Individual tasks running

import tensorflow as tf
import numpy as np
from generate_data1 import CopyTaskData, AssociativeRecallData
from utils import expand, learned_init
from exp3S import Exp3S

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import argparse

parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


########--------------------- NTM ARGUMENTS------------------------------########
#made some changes to fit my task
parser.add_argument('--mann', type=str, default='ntm', help='none | ntm')
parser.add_argument('--num_layers', type=int, default=1) #changing LSTM layers
parser.add_argument('--num_units', type=int, default=10) #changing number of LSTMcell unit
# MEMORY SIZE 128 x 20
parser.add_argument('--num_memory_locations', type=int, default=128)
parser.add_argument('--memory_size', type=int, default=20)
#Number of heads
parser.add_argument('--num_read_heads', type=int, default=1)
parser.add_argument('--num_write_heads', type=int, default=1)

parser.add_argument('--conv_shift_range', type=int, default=1, help='only necessary for ntm')
parser.add_argument('--clip_value', type=int, default=20, help='Maximum absolute value of controller and outputs.')
parser.add_argument('--init_mode', type=str, default='learned', help='learned | constant | random')

parser.add_argument('--optimizer', type=str, default='Adam', help='RMSProp | Adam') # Adam optimizer
parser.add_argument('--learning_rate', type=float, default=0.001)            #Learning Rate
parser.add_argument('--max_grad_norm', type=float, default=50)               #Used to control gradient explosion
parser.add_argument('--num_train_steps', type=int, default=10000)            #Iteration times
parser.add_argument('--batch_size', type=int, default=32)                    #Batch size
parser.add_argument('--eval_batch_size', type=int, default=640)              #Batch size for evaluation

parser.add_argument('--curriculum', type=str, default='none', help='none | uniform | naive | look_back | look_back_and_forward | prediction_gain')
parser.add_argument('--pad_to_max_seq_len', type=str2bool, default=False)

parser.add_argument('--task', type=str, default='copy', help='copy | associative_recall')
parser.add_argument('--num_bits_per_vector', type=int, default=8)            #Features of 1 single input number
parser.add_argument('--max_seq_len', type=int, default=20)                   #Input Sequence Length

parser.add_argument('--verbose', type=str2bool, default=True, help='if true prints lots of feedback')
parser.add_argument('--experiment_name', type=str, default='copy_ntm')
parser.add_argument('--job-dir', type=str, required=False)
parser.add_argument('--steps_per_eval', type=int, default=200)
parser.add_argument('--use_local_impl', type=str2bool, default=True, help='whether to use the repos local NTM implementation or the TF contrib version')

args = parser.parse_args()

if args.mann == 'ntm':
    if args.use_local_impl:
        from ntm import NTMCell
    else:
        from tensorflow.contrib.rnn.python.ops.rnn_cell import NTMCell

if args.verbose:
    import pickle
    HEAD_LOG_FILE = 'head_logs/{0}.p'.format(args.experiment_name)
    GENERALIZATION_HEAD_LOG_FILE = 'head_logs/generalization_{0}.p'.format(args.experiment_name)

class BuildModel(object):
    def __init__(self, max_seq_len, inputs):
        self.max_seq_len = max_seq_len
        self.inputs = inputs
        self._build_model()

    def _build_model(self):
        if args.mann == 'none':
            def single_cell(num_units):
                return tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0)

            cell = tf.contrib.rnn.OutputProjectionWrapper(
                tf.contrib.rnn.MultiRNNCell([single_cell(args.num_units) for _ in range(args.num_layers)]),
                args.num_bits_per_vector,
                activation=None)

            initial_state = tuple(tf.contrib.rnn.LSTMStateTuple(
                c=expand(tf.tanh(learned_init(args.num_units)), dim=0, N=args.batch_size),
                h=expand(tf.tanh(learned_init(args.num_units)), dim=0, N=args.batch_size))
                for _ in range(args.num_layers))

        elif args.mann == 'ntm':
            if args.use_local_impl:
                cell = NTMCell(args.num_layers, args.num_units, args.num_memory_locations, args.memory_size,
                    args.num_read_heads, args.num_write_heads, addressing_mode='content_and_location',
                    shift_range=args.conv_shift_range, reuse=False, output_dim=args.num_bits_per_vector,
                    clip_value=args.clip_value, init_mode=args.init_mode)
            else:
                def single_cell(num_units):
                    return tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0)

                controller = tf.contrib.rnn.MultiRNNCell(
                    [single_cell(args.num_units) for _ in range(args.num_layers)])

                cell = NTMCell(controller, args.num_memory_locations, args.memory_size,
                    args.num_read_heads, args.num_write_heads, shift_range=args.conv_shift_range,
                    output_dim=args.num_bits_per_vector,
                    clip_value=args.clip_value)

        output_sequence, _ = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=self.inputs,
            time_major=False,
            dtype=tf.float32,
            initial_state=initial_state if args.mann == 'none' else None)

        if args.task == 'copy':
            self.output_logits = output_sequence[:, self.max_seq_len+1:, :]
        elif args.task == 'associative_recall':
            self.output_logits = output_sequence[:, 3*(self.max_seq_len+1)+2:, :]

        if args.task in ('copy', 'associative_recall'):
            self.outputs = tf.sigmoid(self.output_logits)

class BuildTModel(BuildModel):
    def __init__(self, max_seq_len, inputs, outputs):
        super(BuildTModel, self).__init__(max_seq_len, inputs)

        if args.task in ('copy', 'associative_recall'):
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=outputs, logits=self.output_logits)
            self.loss = tf.reduce_sum(cross_entropy)/args.batch_size

        if args.optimizer == 'RMSProp':
            optimizer = tf.train.RMSPropOptimizer(args.learning_rate, momentum=0.9, decay=0.9)
        elif args.optimizer == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_variables), args.max_grad_norm)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

with tf.variable_scope('root'):
    max_seq_len_placeholder = tf.placeholder(tf.int32)
    inputs_placeholder = tf.placeholder(tf.float32, shape=(args.batch_size, None, args.num_bits_per_vector+1))
    outputs_placeholder = tf.placeholder(tf.float32, shape=(args.batch_size, None, args.num_bits_per_vector))
    model = BuildTModel(max_seq_len_placeholder, inputs_placeholder, outputs_placeholder)
    initializer = tf.global_variables_initializer()

# training

convergence_on_target_task = None
convergence_on_multi_task = None
performance_on_target_task = None
performance_on_multi_task = None
generalization_from_target_task = None
generalization_from_multi_task = None
if args.task == 'copy':
    data_generator = CopyTaskData()
    target_point = args.max_seq_len
    curriculum_point = 1 if args.curriculum not in ('prediction_gain', 'none') else target_point
    progress_error = 1.0
    convergence_error = 0.1

    if args.curriculum == 'prediction_gain':
        exp3s = Exp3S(args.max_seq_len, 0.001, 0, 0.05)
elif args.task == 'associative_recall':
    data_generator = AssociativeRecallData()
    target_point = args.max_seq_len
    curriculum_point = 2 if args.curriculum not in ('prediction_gain', 'none') else target_point
    progress_error = 1.0
    convergence_error = 0.1

    if args.curriculum == 'prediction_gain':
        exp3s = Exp3S(args.max_seq_len-1, 0.001, 0, 0.05)

###################Written by me#############################
#For counting the number of trainable parameters
def count():
    from functools import reduce
    from operator import mul
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    print("total number of trainable parameters " + str(num_params))

#Edited a little, using gpu to train
with tf.device('/gpu:0'):
    sess = tf.Session()
    sess.run(initializer)
    count()

if args.verbose:
    pickle.dump({target_point: []}, open(HEAD_LOG_FILE, "wb"))
    pickle.dump({}, open(GENERALIZATION_HEAD_LOG_FILE, "wb"))

#Written by me
#For writing loss into files and drawing the plot with files
def write_data(mode,data):
    filename = "head_logs/"+args.experiment_name+"_"+mode+".txt"
    with open(filename, 'a') as f:
        f.write(str(data)+"\n")

def run_eval(batches, store_heat_maps=False, generalization_num=None):
    task_loss = 0
    task_error = 0
    num_batches = len(batches)
    for seq_len, inputs, labels in batches:
        task_loss_, outputs = sess.run([model.loss, model.outputs],
            feed_dict={
                inputs_placeholder: inputs,
                outputs_placeholder: labels,
                max_seq_len_placeholder: seq_len
            })

        task_loss += task_loss_
        task_error += data_generator.error_per_seq(labels, outputs, args.batch_size)

    if store_heat_maps:
        if generalization_num is None:
            tmp = pickle.load(open(HEAD_LOG_FILE, "rb"))
            tmp[target_point].append({
                'labels': labels[0],
                'outputs': outputs[0],
                'inputs': inputs[0]
            })
            pickle.dump(tmp, open(HEAD_LOG_FILE, "wb"))
        else:
            tmp = pickle.load(open(GENERALIZATION_HEAD_LOG_FILE, "rb"))
            if tmp.get(generalization_num) is None:
                tmp[generalization_num] = []
            tmp[generalization_num].append({
                'labels': labels[0],
                'outputs': outputs[0],
                'inputs': inputs[0]
            })
            pickle.dump(tmp, open(GENERALIZATION_HEAD_LOG_FILE, "wb"))


    task_loss /= float(num_batches)
    task_error /= float(num_batches)
    return task_loss, task_error

def eval_performance(curriculum_point, store_heat_maps=False):
    # target task
    batches = data_generator.generate_batches(
        int(int(args.eval_batch_size/2)/args.batch_size),
        args.batch_size,
        bits_per_vector=args.num_bits_per_vector,
        curriculum_point=None,
        max_seq_len=args.max_seq_len,
        curriculum='none',
        pad_to_max_seq_len=args.pad_to_max_seq_len
    )

    target_task_loss, target_task_error = run_eval(batches, store_heat_maps=store_heat_maps)

    # multi-task

    batches = data_generator.generate_batches(
        int(args.eval_batch_size/args.batch_size),
        args.batch_size,
        bits_per_vector=args.num_bits_per_vector,
        curriculum_point=None,
        max_seq_len=args.max_seq_len,
        curriculum='deterministic_uniform',
        pad_to_max_seq_len=args.pad_to_max_seq_len
    )

    multi_task_loss, multi_task_error = run_eval(batches)

    # curriculum point
    if curriculum_point is not None:
        batches = data_generator.generate_batches(
            int(int(args.eval_batch_size/4)/args.batch_size),
            args.batch_size,
            bits_per_vector=args.num_bits_per_vector,
            curriculum_point=curriculum_point,
            max_seq_len=args.max_seq_len,
            curriculum='naive',
            pad_to_max_seq_len=args.pad_to_max_seq_len
        )

        curriculum_point_loss, curriculum_point_error = run_eval(batches)
    else:
        curriculum_point_error = curriculum_point_loss = None

    return target_task_error, target_task_loss, multi_task_error, multi_task_loss, curriculum_point_error, curriculum_point_loss

def eval_generalization():
    res = []
    los = []
    if args.task == 'copy':
        seq_lens = [40, 60, 80, 100, 120]
    elif args.task == 'associative_recall':
        seq_lens = [7, 8, 9, 10, 11, 12]

    for i in seq_lens:
        batches = data_generator.generate_batches(
            6,
            args.batch_size,
            bits_per_vector=args.num_bits_per_vector,
            curriculum_point=i,
            max_seq_len=args.max_seq_len,
            curriculum='naive',
            pad_to_max_seq_len=False
        )

        loss, error = run_eval(batches, store_heat_maps=args.verbose, generalization_num=i)
        res.append(error)
        los.append(loss)
    return res,los

for i in range(args.num_train_steps):
    if args.curriculum == 'prediction_gain':
        if args.task == 'copy':
            task = 1 + exp3s.draw_task()
        elif args.task == 'associative_recall':
            task = 2 + exp3s.draw_task()

    seq_len, inputs, labels = data_generator.generate_batches(
        1,
        args.batch_size,
        bits_per_vector=args.num_bits_per_vector,
        curriculum_point=curriculum_point if args.curriculum != 'prediction_gain' else task,
        max_seq_len=args.max_seq_len,
        curriculum=args.curriculum,
        pad_to_max_seq_len=args.pad_to_max_seq_len
    )[0]

    train_loss, _, outputs = sess.run([model.loss, model.train_op, model.outputs],
        feed_dict={
            inputs_placeholder: inputs,
            outputs_placeholder: labels,
            max_seq_len_placeholder: seq_len
        })

    if args.curriculum == 'prediction_gain':
        loss, _ = run_eval([(seq_len, inputs, labels)])
        v = train_loss - loss
        exp3s.update_w(v, seq_len)

    avg_errors_per_seq = data_generator.error_per_seq(labels, outputs, args.batch_size)

    if args.verbose:
        logger.info('Train loss ({0}): {1}'.format(i, train_loss))
        write_data("train",train_loss)
        logger.info('curriculum_point: {0}'.format(curriculum_point))
        logger.info('Average errors/sequence: {0}'.format(avg_errors_per_seq))
        logger.info('TRAIN_PARSABLE: {0},{1},{2},{3}'.format(i, curriculum_point, train_loss, avg_errors_per_seq))

    if i % args.steps_per_eval == 0:
        target_task_error, target_task_loss, multi_task_error, multi_task_loss, curriculum_point_error, \
        curriculum_point_loss = eval_performance(curriculum_point if args.curriculum != 'prediction_gain' else None, store_heat_maps=args.verbose)

        if convergence_on_multi_task is None and multi_task_error < convergence_error:
            convergence_on_multi_task = i

        if convergence_on_target_task is None and target_task_error < convergence_error:
            convergence_on_target_task = i

        gen_evaled = False
        los = []
        if convergence_on_multi_task is not None and (performance_on_multi_task is None or multi_task_error < performance_on_multi_task):
            performance_on_multi_task = multi_task_error
            generalization_from_multi_task,los = eval_generalization()
            gen_evaled = True

        if convergence_on_target_task is not None and (performance_on_target_task is None or target_task_error < performance_on_target_task):
            performance_on_target_task = target_task_error
            if gen_evaled:
                generalization_from_target_task = generalization_from_multi_task
            else:
                generalization_from_target_task,los = eval_generalization()

        if curriculum_point_error < progress_error:
            if args.task == 'copy':
                curriculum_point = min(target_point, 2 * curriculum_point)
            elif args.task == 'associative_recall':
                curriculum_point = min(target_point, curriculum_point+1)
#########Write LOSS into file##################################################
        p,los = eval_generalization()

        logger.info('----EVAL----')
        logger.info('target task error/loss: {0},{1}'.format(target_task_error, target_task_loss))
        ######Here#####
        write_data("test_same",target_task_loss)
        logger.info('multi task error/loss: {0},{1}'.format(multi_task_error, multi_task_loss))
        logger.info('curriculum point error/loss ({0}): {1},{2}'.format(curriculum_point, curriculum_point_error, curriculum_point_loss))
        logger.info('EVAL_PARSABLE: {0},{1},{2},{3},{4},{5},{6},{7}'.format(i, target_task_error, target_task_loss,
            multi_task_error, multi_task_loss, curriculum_point, curriculum_point_error, curriculum_point_loss))
        if len(los) != 0:
            if args.task == 'copy':
                name_list = [40, 60, 80, 100, 120]
            elif args.task == 'associative_recall':
                name_list = [7, 8, 9, 10, 11, 12]
            for i in range(len(los)):
                write_data("test_different_"+str(name_list[i]),los[i])

if convergence_on_multi_task is None:
    performance_on_multi_task = multi_task_error
    generalization_from_multi_task,los = eval_generalization()

if convergence_on_target_task is None:
    performance_on_target_task = target_task_error
    generalization_from_target_task,los = eval_generalization()

logger.info('----SUMMARY----')
logger.info('convergence_on_target_task: {0}'.format(convergence_on_target_task))
logger.info('performance_on_target_task: {0}'.format(performance_on_target_task))
logger.info('convergence_on_multi_task: {0}'.format(convergence_on_multi_task))
logger.info('performance_on_multi_task: {0}'.format(performance_on_multi_task))

logger.info('SUMMARY_PARSABLE: {0},{1},{2},{3}'.format(convergence_on_target_task, performance_on_target_task,
            convergence_on_multi_task, performance_on_multi_task))

logger.info('generalization_from_target_task: {0}'.format(','.join(map(str, generalization_from_target_task)) if generalization_from_target_task is not None else None))
logger.info('generalization_from_multi_task: {0}'.format(','.join(map(str, generalization_from_multi_task)) if generalization_from_multi_task is not None else None))
