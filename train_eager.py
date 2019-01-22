import tensorflow as tf
import numpy as np
from tensorflow.python import keras
import time

BATCH_SIZE = 256
EPOCHS = 1
TOTAL_TRAIN = 60000
TOTAL_TEST = 10000


def input_fn(images, labels, epochs, batch_size, trainable=True):
    # Convert the inputs to a Dataset. (E)
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.map(lambda images, labels: (images / 255.0, labels), num_parallel_calls=6)  # preprocess image
    if trainable:
        # Shuffle, repeat, and batch the examples. (T)
        SHUFFLE_SIZE = images.shape[0]
        ds = ds.shuffle(SHUFFLE_SIZE)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(batch_size)
        ds = ds.repeat(epochs)
    else:
        ds = ds.batch(1)

    return ds


class MyModel(keras.Model):
    def __init__(self, trainable):
        super(MyModel, self).__init__()
        self.trainable = trainable
        self.n_classes = 10
        # self.relu = keras.activations.relu
        # self.softmax = keras.activations.softmax
        #
        # self.maxpool = keras.layers.MaxPool2D([2, 2], (2, 2), name='maxpool')
        # self.flatten = keras.layers.Flatten(name='flatten')
        # if self.trainable is True:
        #     self.dropout = keras.layers.Dropout(0.5, name='dropout')
        #
        # self.conv1 = keras.layers.Conv2D(16, [5, 5], (1, 1), 'same', name='conv1')
        # self.batchnorm1 = keras.layers.BatchNormalization(trainable=self.trainable, name='batchnorm1')
        # self.conv2 = keras.layers.Conv2D(32, [3, 3], (1, 1), 'same', name='conv2')
        # self.batchnorm2 = keras.layers.BatchNormalization(trainable=self.trainable, name='batchnorm2')
        # self.dense1 = keras.layers.Dense(128, name='dense1')
        # self.dense2 = keras.layers.Dense(self.n_classes, name='dense2')
        self.flatten = keras.layers.Flatten(name='flatten')
        self.dense = keras.layers.Dense(self.n_classes, name='dense')

    @tf.function
    def call(self, inputs):
        # x = self.conv1(inputs)
        # x = self.batchnorm1(x)
        # x = self.relu(x, max_value=6)
        # x = self.maxpool(x)
        # x = self.conv2(x)
        # x = self.batchnorm2(x)
        # x = self.relu(x, max_value=6)
        # x = self.maxpool(x)
        # x = self.flatten(x)
        # x = self.dense1(x)
        # if self.trainable is True:
        #     x = self.dropout(x)
        # x = self.dense2(x)
        # x = self.softmax(x)
        x = self.flatten(inputs)
        x = self.dense(x)
        return x


def loss(model, logits, labels):
    # arg_max = tf.argmax(labels, axis=1)
    loss = keras.losses.sparse_categorical_crossentropy(
        y_true=labels,
        y_pred=logits
    )
    reg_losses = [keras.regularizers.l2(1e-4)(w) for w in model.trainable_weights]

    loss = loss + tf.add_n(reg_losses)

    mean_loss = tf.reduce_mean(loss, name='loss')

    return mean_loss


def acc(logits, labels):
    acc = keras.metrics.sparse_categorical_accuracy(
        y_true=labels,
        y_pred=logits
    )
    acc = tf.reduce_mean(acc, name='acc')

    return acc


def grad(model, image, label):
    with tf.GradientTape() as tape:

        logits = model(image)
        loss_value = loss(model, logits, label)
        acc_value = acc(logits, label)

    return tape.gradient(loss_value, model.trainable_variables), loss_value, acc_value


def train(model, datasets, log_freq=10):

    optimizer = tf.keras.optimizers.Adam(lr=1e-4)
    global_step = optimizer.iterations
    ckpt = tf.train.Checkpoint(step=global_step, optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, './logs', max_to_keep=1)

    for image, label in datasets:
        grads, loss_value, acc_value = grad(model, image, label)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if global_step.numpy() % (TOTAL_TRAIN // BATCH_SIZE) == 0:
            save_path = manager.save(checkpoint_number=global_step.numpy())
            print("Saved checkpoint for step {}: {}".format(global_step.numpy(), save_path))
        if global_step.numpy() % log_freq == 0:
            print('step = ', global_step.numpy(), '\t', 'loss = {:.4f}'.format(loss_value.numpy()),
                  '\t', 'acc = {:.4f}'.format(acc_value.numpy()))

    manager.save(checkpoint_number=global_step.numpy())
    print(manager.latest_checkpoint)


def main():

    train_x, train_y = np.load('./mnist_datasets/x_train.npy'), np.load('./mnist_datasets/y_train.npy')
    train_x = np.expand_dims(train_x, axis=-1).astype(np.float32)

    # def make_one_hot(data):
    #     return (np.arange(10) == data[:, None]).astype(np.uint8)
    #
    #
    # train_y = make_one_hot(train_y)

    print(train_x.shape, '\t', train_y.shape)

    model = MyModel(trainable=True)
    dataset = input_fn(train_x, train_y, EPOCHS, BATCH_SIZE, trainable=True)

    #  get model structure by one inference
    #  build model for model.*
    model(tf.zeros([1, 28, 28, 1]))
    print('train vatiables...')
    for train_i in model.trainable_variables:
        print(train_i.name)
    print('not train vatiables...')
    for not_train_i in model.non_trainable_variables:
        print(not_train_i.name)

    print(model.inputs, model.outputs)  #
    model.summary()

    start = time.time()
    train(model, dataset)
    end = time.time()

    print('train_time: {} ms'.format((end-start)*1000))

    tf.saved_model.save(model, './pb_model/saved_model', signatures=model.call.get_concrete_function(
        tf.TensorSpec(shape=[None, 28, 28, 1], dtype=tf.float32, name='input_tensor')))
    print('saved')


if __name__ == '__main__':
    main()
