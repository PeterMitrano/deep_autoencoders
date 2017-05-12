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


class Model:
    def __init__(self, images, global_step, batch_size):
        self.trainers = []
        self.losses = []

        self.flat_norm_images = tf.reshape(images, [-1, img_dim], name='flatten')

        with tf.name_scope('layer_1'):
            self.h1_dim = 100
            self.w1 = tf.Variable(tf.truncated_normal([img_dim, self.h1_dim], 0.0, 0.1), name='w1')
            self.w1_trans = tf.transpose(self.w1, [1, 0])
            self.b1 = tf.Variable(tf.constant(0.05, shape=[self.h1_dim]), name='b1')
            self.a1 = tf.Variable(tf.constant(0.05, shape=[img_dim]), name='a1')
            self.h1 = tf.nn.sigmoid(tf.matmul(self.flat_norm_images, self.w1) + self.b1, name='h1')
            self.y1 = tf.nn.sigmoid(tf.matmul(self.h1, self.w1_trans) + self.a1, name='y1')
            self.y1_images = tf.reshape(self.y1, [-1, IMAGE_SIZE, IMAGE_SIZE, 3], name='y1_images')
            self.vars1 = [self.w1, self.b1, self.a1]

            self.loss1 = tf.nn.l2_loss(self.y1 - self.flat_norm_images, name='loss1')
            self.train1 = tf.train.AdamOptimizer(0.002).minimize(self.loss1, global_step, self.vars1, name='train1')
            self.losses.append(self.loss1)
            self.trainers.append(self.train1)

            tf.summary.scalar('loss1', self.loss1)
            tf.summary.histogram('w1', self.w1)
            tf.summary.histogram('b1', self.b1)
            tf.summary.histogram('a1', self.a1)
            tf.summary.histogram('h1', self.h1)
            tf.summary.histogram('y1', self.y1)
            tf.summary.image('y1_images', self.y1_images, max_outputs=10)

        tf.summary.image('images', images, max_outputs=10)
        tf.summary.histogram('images', self.flat_norm_images)


def main():
    batch_size = 128
    data_dir = 'cifar'
    train_filenames = [os.path.join(data_dir, 'data_batch_%i.bin' % i) for i in range(1, 6)]

    filename_queue = tf.train.string_input_producer(train_filenames)

    with tf.name_scope("read"):
        reader = tf.FixedLengthRecordReader(record_bytes=img_dim + 1, name='input_reader')
        _, record_str = reader.read(filename_queue, name='read_op')
        record_raw = tf.decode_raw(record_str, tf.uint8, name='decode_raw')

        label = tf.cast(tf.slice(record_raw, [0], [1]), tf.int32)
        image = tf.reshape(tf.slice(record_raw, [1], [img_dim]), [3, 32, 32])
        float_image = tf.cast(tf.transpose(image, [1, 2, 0]), tf.float32)
        image = tf.divide(float_image, 255.0, name='norm_images')

    with tf.name_scope('preprocess'):
        reshaped_image = tf.cast(image, tf.float32)
        # float_image = tf.image.per_image_standardization(reshaped_image)
        processed_image = reshaped_image

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)

    images, labels = generate_image_and_label_batch(processed_image, label, min_queue_examples, batch_size)

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

    m = Model(images, global_step, batch_size)

    init = tf.global_variables_initializer()
    summaries = tf.summary.merge_all()

    # Open text editor to write description of the run and commit it
    if '--temp' not in sys.argv:
        cmd = ['git', 'commit', __file__]
        os.environ['TF_LOG_DIR'] = log_path
        call(cmd)

    writer.add_graph(sess.graph)

    sess.run(init)

    layer_schedule = [4000]
    layer = 0
    layer_it = 0
    for i in range(4000):
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
