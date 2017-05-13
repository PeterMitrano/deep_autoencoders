#!/usr/bin/python3.5

import tensorflow as tf
import numpy as np
import sys
from subprocess import call
from datetime import datetime
import os

# Global constants describing the CIFAR-10 data set.
N_CHANNELS = 1
IMAGE_SIZE = 32
img_dim = IMAGE_SIZE * IMAGE_SIZE * N_CHANNELS
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

        return images, tf.one_hot(tf.squeeze(label_batch), depth=NUM_CLASSES)


class Model:
    def __init__(self, image_batch, label_batch, global_step, batch_size):
        self.trainers = []
        self.losses = []

        self.image_batch_flat = tf.reshape(image_batch, [-1, img_dim], name='flatten')

        with tf.name_scope('layer_1'):
            self.h1_dim = 500
            self.w1 = tf.Variable(tf.truncated_normal([img_dim, self.h1_dim], 0.0, 0.1), name='w1')
            self.w1_trans = tf.transpose(self.w1, [1, 0])
            self.w1_viz = tf.reshape(self.w1_trans, [-1, IMAGE_SIZE, IMAGE_SIZE, N_CHANNELS], name='w1_viz')
            self.b1 = tf.Variable(tf.constant(0.05, shape=[self.h1_dim]), name='b1')
            self.a1 = tf.Variable(tf.constant(0.05, shape=[img_dim]), name='a1')
            self.h1 = tf.nn.sigmoid(tf.matmul(self.image_batch_flat, self.w1) + self.b1, name='h1')
            self.y1 = tf.nn.sigmoid(tf.matmul(self.h1, self.w1_trans) + self.a1, name='y1')
            self.y1_image_batch = tf.reshape(self.y1, [-1, IMAGE_SIZE, IMAGE_SIZE, N_CHANNELS], name='y1_images')
            self.vars1 = [self.w1, self.b1, self.a1]

            self.reconstruction_loss1 = tf.nn.l2_loss(self.y1 - self.image_batch_flat, name='reconstruction_loss1')
            self.loss1 = tf.nn.l2_loss(self.y1 - self.image_batch_flat, name='loss1')
            self.train1 = tf.train.AdamOptimizer(0.002).minimize(self.loss1, global_step, self.vars1, name='train1')
            self.losses.append(self.loss1)
            self.trainers.append(self.train1)

            tf.summary.scalar('reconstruction_loss1', self.reconstruction_loss1)
            # tf.summary.histogram('w1', self.w1)
            # tf.summary.histogram('b1', self.b1)
            # tf.summary.histogram('a1', self.a1)
            # tf.summary.histogram('h1', self.h1)
            # tf.summary.histogram('y1', self.y1)
            # tf.summary.image('w1_viz', self.w1_viz, max_outputs=100)
            tf.summary.image('y1_images', self.y1_image_batch, max_outputs=10)

        with tf.name_scope('layer_2'):
            self.h2_dim = 100
            self.w2 = tf.Variable(tf.truncated_normal([self.h1_dim, self.h2_dim], 0.0, 0.1), name='w2')
            self.w2_trans = tf.transpose(self.w2, [1, 0])
            self.b2 = tf.Variable(tf.constant(0.05, shape=[self.h2_dim]), name='b2')
            self.a2 = tf.Variable(tf.constant(0.05, shape=[self.h1_dim]), name='a2')
            self.h2 = tf.nn.sigmoid(tf.matmul(self.h1, self.w2) + self.b2, name='h2')
            self.h1_ = tf.nn.sigmoid(tf.matmul(self.h2, self.w2_trans) + self.a2, name='h2_')
            self.y2 = tf.nn.sigmoid(tf.matmul(self.h1_, self.w1_trans) + self.a1, name='y2')
            self.y2_image_batch = tf.reshape(self.y2, [-1, IMAGE_SIZE, IMAGE_SIZE, N_CHANNELS], name='y2_images')
            self.vars2 = [self.w2, self.b2, self.a2]

            self.reconstruction_loss2 = tf.nn.l2_loss(self.y2 - self.image_batch_flat, name='reconstruction_loss2')
            self.loss2 = tf.nn.l2_loss(self.h1_ - self.h1, name='loss2')
            self.train2 = tf.train.AdamOptimizer(0.002).minimize(self.loss2, global_step, self.vars2, name='train2')
            self.losses.append(self.loss2)
            self.trainers.append(self.train2)

            tf.summary.scalar('reconstruction_loss2', self.reconstruction_loss2)
            # tf.summary.histogram('w2', self.w2)
            # tf.summary.histogram('b2', self.b2)
            # tf.summary.histogram('a2', self.a2)
            # tf.summary.histogram('h2', self.h2)
            # tf.summary.histogram('y2', self.y2)
            tf.summary.image('y2_images', self.y1_image_batch, max_outputs=10)

        with tf.name_scope("layer_3"):
            self.w3 = tf.Variable(tf.truncated_normal([self.h2_dim, NUM_CLASSES], 0.0, 0.1), name='w3')
            self.b3 = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]), name='b3')
            self.y3 = tf.matmul(self.h2, self.w3) + self.b3
            self.pred = tf.argmax(self.y3, axis=1)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, tf.argmax(label_batch, axis=1)), dtype=tf.float32))
            self.vars3 = [self.w3, self.b3]

            self.loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y3, labels=label_batch), name='classification_loss')
            self.train3 = tf.train.AdamOptimizer(0.001).minimize(self.loss3, global_step, self.vars3, name='train3')
            self.losses.append(self.loss3)
            self.trainers.append(self.train3)

            tf.summary.scalar('classification_loss', self.loss3)
            tf.summary.scalar('accuracy', self.accuracy)

        tf.summary.image('images', image_batch, max_outputs=10)


def main():
    batch_size = 128
    data_dir = 'cifar'
    train_filenames = [os.path.join(data_dir, 'data_batch_%i.bin' % i) for i in range(1, 6)]

    filename_queue = tf.train.string_input_producer(train_filenames)
    raw_img_dim = IMAGE_SIZE * IMAGE_SIZE * 3

    with tf.name_scope("read"):
        reader = tf.FixedLengthRecordReader(record_bytes=raw_img_dim + 1, name='input_reader')
        _, record_str = reader.read(filename_queue, name='read_op')
        record_raw = tf.decode_raw(record_str, tf.uint8, name='decode_raw')

        label = tf.cast(tf.slice(record_raw, [0], [1]), tf.int32)
        image = tf.reshape(tf.slice(record_raw, [1], [raw_img_dim]), [3, 32, 32])
        float_image = tf.cast(tf.transpose(image, [1, 2, 0]), tf.float32)
        norm_image = tf.divide(float_image, 255.0, name='norm_images')

    with tf.name_scope('preprocess'):
        # float_image = tf.image.per_image_standardization(float_image)
        gray_image = tf.reduce_sum(tf.multiply(norm_image, tf.constant([0.2126, 0.7512, 0.0722])), axis=2, name='grayscale')
        gray_image = tf.expand_dims(gray_image, axis=2)
        processed_image = gray_image
        # print(norm_image.get_shape())

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)

    image_batch, label_batch = generate_image_and_label_batch(processed_image, label, min_queue_examples, batch_size)

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

    m = Model(image_batch, label_batch, global_step, batch_size)

    init = tf.global_variables_initializer()
    summaries = tf.summary.merge_all()

    # Open text editor to write description of the run and commit it
    if '--temp' not in sys.argv:
        cmd = ['git', 'commit', __file__]
        os.environ['TF_LOG_DIR'] = log_path
        call(cmd)

    writer.add_graph(sess.graph)

    sess.run(init)

    layer_schedule = [4000, 4000, 2000]
    layer = 0
    layer_it = 0
    for i in range(10000):
        if layer_it == layer_schedule[layer]:
            layer += 1
            layer_it = 0

        train_op = m.trainers[layer]
        loss_op = m.losses[layer]
        layer_it += 1

        sess.run(train_op)

        if i % 10 == 0:
            loss, sum, step = sess.run([loss_op, summaries, global_step])
            writer.add_summary(sum, step)
            print(loss, loss_op)

        if i % 1000 == 0:
            w1_viz = sess.run(m.w1_viz)
            np.save(os.path.join(log_path, 'w1_viz_%i.npy' % i), w1_viz)


if __name__ == '__main__':
    main()
