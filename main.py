import tensorflow as tf
import sys
from subprocess import call
from datetime import datetime
import os

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = 32
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000


def read_cifar10(filename_queue):
    class CifarRecord(object):
        def __init__(self):
            self.image = None
            self.label = None

    result = CifarRecord()

    reader = tf.FixedLengthRecordReader(record_bytes=32 * 32 * 3 + 1, name='input_reader')
    result.key, record_str = reader.read(filename_queue)
    record_raw = tf.decode_raw(record_str, tf.uint8)

    result.label = tf.cast(tf.slice(record_raw, [0], [1]), tf.int32)
    image = tf.reshape(tf.slice(record_raw, [1], [32 * 32 * 3]), [3, 32, 32])
    result.image = tf.cast(tf.transpose(image, [1, 2, 0]), tf.float32)

    return result


def generate_image_and_label_batch(image, label, min_queue_examples, batch_size):
    num_preprocess_threads = 16
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)

    # Display the training images in the visualizer.
    tf.summary.image('images', images)

    return images, tf.reshape(label_batch, [batch_size])


def inputs():
    data_dir = 'cifar'
    train_filenames = [os.path.join(data_dir, 'data_batch_%i.bin' % i) for i in range(5)]

    filename_queue = tf.train.string_input_producer(train_filenames)
    read_input = read_cifar10(filename_queue)

    reshaped_image = tf.cast(read_input.image, tf.float32)

    float_image = tf.image.per_image_standardization(reshaped_image)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)

    batch_size = 256
    return generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size)


def main():
    images, labels = inputs()

    sess = tf.Session()

    init = tf.global_variables_initializer()
    summaries = tf.summary.merge_all()

    tf.train.start_queue_runners()

    day_str = "{:%B_%d}".format(datetime.now())
    time_str = "{:%H:%M:%S}".format(datetime.now())
    day_dir = "log_data/" + day_str + "/"
    log_path = day_dir + day_str + "_" + time_str + "/"
    if not os.path.exists(day_dir):
        os.mkdir(day_dir)

    writer = tf.summary.FileWriter(log_path)

    # Open text editor to write description of the run and commit it
    if '--temp' not in sys.argv:
        cmd = ['git', 'commit', __file__]
        os.environ['TF_LOG_DIR'] = log_path
        call(cmd)

    writer.add_graph(sess.graph)
    sess.run(init)

    global_step = tf.Variable(0, trainable=False, name='global_step')

    sum, step = sess.run([summaries, global_step])
    writer.add_summary(sum, step)


if __name__ == '__main__':
    main()
