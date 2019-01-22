import tensorflow as tf
import numpy as np

from train_eager import MyModel, input_fn, acc


def main():
    # test
    test_x, test_y = np.load('./mnist_datasets/x_test.npy'), np.load('./mnist_datasets/y_test.npy')
    test_x = np.expand_dims(test_x, axis=-1).astype(np.float32)

    # def make_one_hot(data):
    #     return (np.arange(10) == data[:, None]).astype(np.uint8)
    #
    #
    # train_y = make_one_hot(train_y)

    test_x, test_y = test_x, test_y

    print(test_x.shape, '\t', test_y.shape)

    model = MyModel(trainable=False)
    dataset = input_fn(test_x, test_y, 1, 1, trainable=False)
    ckpt = tf.train.Checkpoint(net=model)
    manager = tf.train.CheckpointManager(ckpt, './logs', max_to_keep=1)
    ckpt.restore(manager.latest_checkpoint)

    #  build model for model.*
    logits = model(tf.zeros([1, 28, 28, 1]))
    print('train vatiables...')
    for train_i in model.trainable_variables:
        print(train_i.name)
    print('not train vatiables...')
    for not_train_i in model.non_trainable_variables:
        print(not_train_i.name)

    model.summary()

    num = 0
    res = 0
    for image, label in dataset:
        logits = model(image)
        acc_value = acc(logits, label)
        res += acc_value.numpy()
        num += 1
        if num % 1000 == 0:
            print('step = ', num)

    print(res / num)


if __name__ == '__main__':
    main()
