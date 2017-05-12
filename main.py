#!/usr/bin/python3.5

import tensorflow as tf
import sys
from subprocess import call
from datetime import datetime
import os

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = 32
img_dim = IMAGE_SIZE * IMAGE_SIZE * 3
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000


def read_cifar10(filename_queue):
    with tf.name_scope("read"):
        class CifarRecord(object):
            def __init__(self):
                self.image = None
                self.label = None

        result = CifarRecord()

        reader = tf.FixedLengthRecordReader(record_bytes= img_dim + 1, name='input_reader')
        result.key, record_str = reader.read(filename_queue, name='read_op')
        record_raw = tf.decode_raw(record_str, tf.uint8, name='decode_raw')

        result.label = tf.cast(tf.slice(record_raw, [0], [1]), tf.int32)
        image = tf.reshape(tf.slice(record_raw, [1], [img_dim]), [3, 32, 32])
        result.image = tf.cast(tf.transpose(image, [1, 2, 0]), tf.float32)

        return result


def generate_image_and_label_batch(image, label, min_queue_examples, batch_size):
    with tf.name_scope("make_batches"):
        num_preprocess_threads = 2
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)

        tf.summary.image('images', images)

        return images, tf.reshape(label_batch, [batch_size])


def read_inputs():
    data_dir = 'cifar'
    train_filenames = [os.path.join(data_dir, 'data_batch_%i.bin' % i) for i in range(1, 6)]

    filename_queue = tf.train.string_input_producer(train_filenames)
    read_input = read_cifar10(filename_queue)

    with tf.name_scope('preprocess'):
        reshaped_image = tf.cast(read_input.image, tf.float32)
        float_image = tf.image.per_image_standardization(reshaped_image)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)

    batch_size = 128
    return generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size)


class Model:
    def __init__(self, images, global_step):
        self.trainers = []
        self.losses = []

        images = images / 255.0
        flat_norm_images = tf.reshape(images, [-1, img_dim], name='flatten')

        with tf.name_scope('layer_1'):
            self.h1_dim = 100
            self.w1 = tf.Variable(tf.truncated_normal([img_dim, self.h1_dim], 0.0, 0.1), name='w1')
            self.w1_trans = tf.transpose(self.w1, [1, 0])
            self.b1 = tf.Variable(tf.constant(0.05), [self.h1_dim], name='b1')
            self.a1 = tf.Variable(tf.constant(0.05), [img_dim], name='a1')
            self.h1 = tf.nn.sigmoid(tf.matmul(flat_norm_images, self.w1) + self.b1, name='h1')
            self.y1 = tf.nn.sigmoid(tf.matmul(self.h1, self.w1_trans) + self.a1, name='y1')

            self.vars1 = [self.w1, self.b1, self.a1]

            self.loss1 = tf.nn.l2_loss(self.y1 - flat_norm_images, name='loss1')
            self.train1 = tf.train.AdamOptimizer(0.001).minimize(self.loss1, global_step, self.vars1, name='train1')
            self.losses.append(self.loss1)
            self.trainers.append(self.train1)

            tf.summary.scalar('loss1', self.loss1)
            tf.summary.histogram('w1', self.w1)
            tf.summary.histogram('b1', self.b1)
            tf.summary.histogram('a1', self.a1)
            tf.summary.histogram('y1', self.y1)

        # with tf.name_scope('layer_2'):
        #     self.h2_dim = 750
        #     self.w2 = tf.Variable(tf.truncated_normal([self.h1_dim, self.h2_dim], 0.0, 0.1), name='w2')
        #     self.w2_trans = tf.transpose(self.w2, [1, 0])
        #     self.b2 = tf.Variable(tf.constant(0.05), [self.h2_dim], name='b2')
        #     self.a2 = tf.Variable(tf.constant(0.05), [self.h2_dim], name='a2')
        #     self.h2 = tf.nn.sigmoid(tf.matmul(self.h1, self.w2) + self.b2, name='h2')
        #     self.h2_ = tf.nn.sigmoid(tf.matmul(self.h2, self.w2_trans) + self.a2, name='h2_')
        #     self.y2 = tf.nn.sigmoid(tf.matmul(self.h2_, self.w1_trans) + self.a1, name='y2')
        #     self.y2_images = tf.reshape(self.y2, [-1, IMAGE_SIZE, IMAGE_SIZE, 3], name='y2_images')
        #
        #     self.vars2 = [self.w2, self.b2, self.a2]
        #
        #     self.loss2 = tf.nn.l2_loss(self.y2 - flat_norm_images, name='loss2')
        #     self.train2 = tf.train.AdamOptimizer(0.001).minimize(self.loss2, global_step, self.vars2, name='train2')
        #     self.losses.append(self.loss2)
        #     self.trainers.append(self.train2)
        #
        #     tf.summary.scalar('loss2', self.loss2)
        #     tf.summary.histogram('w2', self.w2)
        #     tf.summary.histogram('b2', self.b2)
        #     tf.summary.histogram('a2', self.a2)
        #     tf.summary.histogram('y2', self.y2)
        #     tf.summary.image('y2_images', self.y2_images)

        tf.summary.image('images', images)


def main():
    images, labels = read_inputs()

    sess = tf.Session()

    tf.train.start_queue_runners(sess=sess)

    day_str = "{:%B_%d}".format(datetime.now())
    time_str = "{:%H:%M:%S}".format(datetime.now())
    day_dir = "log_data/" + day_str + "/"
    log_path = day_dir + day_str + "_" + time_str + "/"
    if not os.path.exists(day_dir):
        os.mkdir(day_dir)

    writer = tf.summary.FileWriter(log_path)

    global_step = tf.Variable(0, trainable=False, name='global_step')

    m = Model(images, global_step)

    init = tf.global_variables_initializer()
    summaries = tf.summary.merge_all()

    # Open text editor to write description of the run and commit it
    if '--temp' not in sys.argv:
        cmd = ['git', 'commit', __file__]
        os.environ['TF_LOG_DIR'] = log_path
        call(cmd)

    writer.add_graph(sess.graph)

    sess.run(init)

    layer_schedule = [1000]
    layer = 0
    layer_it = 0
    for i in range(1000):
        if layer_it == layer_schedule[layer]:
            layer += 1
            layer_it = 0

        train_op = m.trainers[layer]
        loss_op = m.losses[layer]

        sess.run(train_op)

        if i % 10 == 0:
            loss, sum, step = sess.run([loss_op, summaries, global_step])
            writer.add_summary(sum, step)
            print(loss, loss_op)


if __name__ == '__main__':
    main()
