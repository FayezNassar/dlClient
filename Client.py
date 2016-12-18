import http.client
import chainer.links as L
import chainer
from MLP import MLP
import paramiko
import re
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset
import numpy as np
import json
import time


lin_neural_network_l1 = L.Linear(784, 300)
lin_neural_network_l2 = L.Linear(300, 10)
mlp = None


def main():
    conn = http.client.HTTPConnection("localhost", 8000)
    conn.request("GET", "/hello/requestWork")
    response = conn.getresponse()
    response_message = response.read().decode('utf-8')
    conn.close()
    image_file_index = json.loads(response_message)['image_file_index']
    mode = str(json.loads(response_message)['mode'])
    l1_w_list = json.loads(response_message)['l1_w']
    l2_w_list = json.loads(response_message)['l2_w']

    for i in range(300):
        for j in range(784):
            lin_neural_network_l1.W.data[i][j] = l1_w_list[i][j]
    for i in range(10):
        for j in range(300):
            lin_neural_network_l2.W.data[i][j] = l2_w_list[i][j]

    global mlp
    mlp = MLP(lin_neural_network_l1, lin_neural_network_l2)
    if (mode == "train"):
        train(image_file_index, l1_w_list, l2_w_list)
    if (mode == "validation"):
        validate()


def train(image_file_index, l1_w_list, l2_w_list):
    print("working in file: " + str(image_file_index))
    file_images_name = '~/images_train/image_' + str(image_file_index)
    file_labels_name = '~/labels_train/label_' + str(image_file_index)
    global mlp
    # Create a model
    model = L.Classifier(mlp)
    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('t2.technion.ac.il', username='sfirasss', password='Firspn3#')
    cmd_images = 'cat ' + file_images_name
    images_stdin, images_stdout, images_stderr = ssh.exec_command(cmd_images)
    read_image = str(images_stdout.read(), 'utf-8')
    read_image = re.split('\[', read_image)
    cmd_labels = 'cat ' + file_labels_name
    labels_stdin, labels_stdout, labels_stderr = ssh.exec_command(cmd_labels)
    read_labels = str(labels_stdout.read(), 'utf-8')
    ssh.close()

    label_array = np.fromstring(read_labels, dtype=np.int32, sep=',')
    images_array = [None] * 1200
    for i in range(0, 1200):
        images_array[i] = np.fromstring(read_image[i + 1], dtype=np.float32, sep=' ')

    train_tuple = tuple_dataset.TupleDataset(images_array, label_array)
    # train_, test = chainer.datasets.get_mnist()
    # TODO: check with roman the mini batch size in the client side.
    train_iter = chainer.iterators.SerialIterator(train_tuple, 1)  # 1 = mini patch

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=-1)
    trainer = training.Trainer(updater, (1, 'epoch'), out='result')

    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.ProgressBar())

    # Run the training
    start_time = time.time()
    trainer.run()
    end_time = time.time()

    # compute the deltas and send them
    total_time = end_time - start_time
    delta_layer1 = np.zeros((300, 784), dtype='float32')
    delta_layer2 = np.zeros((10, 300), dtype='float32')
    for i in range(10):
        for j in range(300):
            delta_layer2[i][j] = mlp.l2.W.data[i][j] - l2_w_list[i][j];
    for i in range(300):
        for j in range(784):
            delta_layer1[i][j] = mlp.l1.W.data[i][j] - l1_w_list[i][j];

    print(type(delta_layer1))
    print(type(delta_layer1.tolist()))
    data = {
        'work_time': total_time,
        # 'l1_delta': delta_layer1.tolist(),
        # 'l2_delta': delta_layer2.tolist()
    }
    print("total_time: " + str(total_time))
    conn = http.client.HTTPConnection("localhost", 8000)
    conn.request("GET", "/hello/responseResult", json.dumps(data))
    conn.close()


def validate():
    return 0


if __name__ == '__main__':
    main()
