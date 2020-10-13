"""Private training on combined data from several data owners"""
import tf_encrypted as tfe
import tensorflow as tf
import pandas as pd
from common import DataOwner, ModelOwner, LinearRegression, DataSchema

from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

import time

# test data set configuration
test_set = [
    {'n': 'default_credit', 'r': 30000, 'f': 24, 's': 23},
    {'n': 'breast', 'r': 569, 'f': 31, 's': 30},
    {'n': 'financial', 'r': 3392, 'f': 225, 's': 224},
    {'n': 'sonar', 'r': 208, 'f': 61, 's': 60}
]


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def forward(w, b, x):
    out = np.dot(x, w) + b
    return sigmoid(out)


def inference(w, b, f_x, f_y):
    xdf = pd.read_csv(f_x, header=None)
    xdf.insert(len(xdf.columns), len(xdf.columns), 0.)
    ydf = pd.read_csv(f_y, header=None)
    ydf.pop(1)

    x = xdf.to_numpy()
    y = ydf.to_numpy()
    y_p = forward(w, b, x)
    # print(f'--> y_pred:{y_p}')
    # print(f'--> y_true:{y}')
    auc = roc_auc_score(y, y_p)
    return auc


def data_reveal(sess, data_owner, data):
    """ Print data value with tf.print """
    # @tfe.local_computation
    def _print(x):
        op = tf.print('Value on {}'.format(data_owner.player_name),
                      x, summarize=-1)
        return op

    reveal_op = tfe.define_output(data_owner.player_name,
                                  [data],
                                  _print)
    sess.run(reveal_op)


def main(server):
    case = test_set[0]
    print('===================================================='
          f'Running {case}:'
          '====================================================')

    name = case['n']
    num_rows = case['r']
    num_features = case['f']
    num_epoch = 30
    batch_size = 1024
    num_batches = (num_rows // batch_size) * num_epoch

    # who shall receive the output
    model_owner = ModelOwner('alice')

    data_schema0 = DataSchema([tf.float64] * case['s'], [0.0] * case['s'])
    # data_schema1 = DataSchema([tf.int64]+[tf.float64]*16, [0]+[0.0]*16)
    data_schema1 = DataSchema([tf.int64] + [tf.float64] * 1, [0] + [0.0] * 1)
    data_owner_0 = DataOwner('alice',
                             f'{name}/{name}_tfe_host.csv',
                             data_schema0,
                             batch_size=batch_size)
    data_owner_1 = DataOwner('bob',
                             f'{name}/{name}_tfe_guest.csv',
                             data_schema1,
                             batch_size=batch_size)

    tfe.set_protocol(tfe.protocol.Pond(
        tfe.get_config().get_player(data_owner_0.player_name),
        tfe.get_config().get_player(data_owner_1.player_name)
    ))

    x_train_0 = tfe.define_private_input(
        data_owner_0.player_name,
        data_owner_0.provide_data
    )
    x_train_1 = tfe.define_private_input(
        data_owner_1.player_name,
        data_owner_1.provide_data
    )
    y_train = tfe.gather(x_train_1, 0, axis=1)
    y_train = tfe.reshape(y_train, [batch_size, 1])

    # Remove bob's first column (which is label)
    x_train_1 = tfe.strided_slice(x_train_1,
                                  [0, 1],
                                  [x_train_1.shape[0], x_train_1.shape[1]],
                                  [1, 1])
    x_train = tfe.concat([x_train_0, x_train_1], axis=1)

    model = LinearRegression(num_features)
    reveal_weights_op = model_owner.receive_weights(model.weights)

    with tfe.Session() as sess:
        sess.run(tfe.global_variables_initializer(), tag='init')
        # sess.run(tf.initialize_local_variables())

        num_batches = num_rows // batch_size
        for epoch in range(num_epoch):
            print(f'Epoch {epoch}:')
            model.fit(sess, x_train, y_train, num_batches)
            # TODO(Morten)
            # each evaluation results in nodes for a forward pass being added to the graph;
            # maybe there's some way to avoid this, even if it means only if the shapes match
            # model.evaluate(sess, x_train, y_train, data_owner_0)
            # model.evaluate(sess, x_train, y_train, data_owner_1)

            plain_w, plain_b = sess.run([model.weights[0].reveal(),
                                         model.weights[1].reveal()])
            # print(f'plain reveal model[w:b]: {plain_w.shape, type(plain_w), plain_b.shape}')
            auc = inference(plain_w, plain_b,
                            f'{name}/{name}_tfe_host.csv',
                            f'{name}/{name}_tfe_guest.csv')
            print(f'=> Train AUC: {auc}')

        # y_all = sess.run(y_train.reveal())
        # y_all = pd.DataFrame(y_all)
        # y_all.to_csv('y_all.csv')

        # print model weights
        # sess.run(reveal_weights_op, tag='reveal')

        # data_reveal(sess, data_owner_0, model.weights)
        # data_reveal(sess, data_owner_0, y_train)
        # data_reveal(sess, data_owner_1, y_train)


def start_master(cluster_config_file=None):
    print("Starting alice...")
    remote_config = tfe.RemoteConfig.load(cluster_config_file)
    tfe.set_config(remote_config)
    tfe.set_protocol(tfe.protocol.Pond())
    players = remote_config.players
    server0 = remote_config.server(players[0].name)

    st = time.perf_counter()
    main(server0)
    ed = time.perf_counter()
    print(f'Elapsed time: {ed - st}s')


if __name__ == "__main__":
    start_master("config.json")
