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
        flat_images = tf.reshape(images, [-1, img_dim], name='flatten')

        with tf.name_scope('layer_1'):
            self.h1_dim = 1000
            self.w1 = tf.Variable(tf.truncated_normal([img_dim, self.h1_dim], 0, 0.1), name='w1')
            self.w1_trans = tf.transpose(self.w1, [1, 0])
            self.b1 = tf.Variable(tf.constant(0.1), [self.h1_dim], name='b1')
            self.a1 = tf.Variable(tf.constant(0.1), [img_dim], name='a1')
            self.z1 = tf.nn.sigmoid(tf.matmul(flat_images, self.w1) + self.b1, name='z1')
            self.y1 = tf.nn.sigmoid(tf.matmul(self.z1, self.w1_trans) + self.a1, name='y1')
            self.loss1 = tf.nn.l2_loss(self.y1 - flat_images, name='loss1')
            self.vars1 = [self.w1, self.b1, self.a1]
            self.train1 = tf.train.AdamOptimizer(0.0001).minimize(self.loss1, global_step, self.vars1, name='train1')

            tf.summary.scalar('loss1', self.loss1)


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

    for i in range(500):
        _, loss1 = sess.run([m.train1, m.loss1])

        if i % 10 == 0:
            print(loss1)
            sum, step = sess.run([summaries, global_step])
            writer.add_summary(sum, step)


if __name__ == '__main__':
    main()
