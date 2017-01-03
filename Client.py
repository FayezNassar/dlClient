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
import requests
import time

url = 'https://deeplearningserver.herokuapp.com/hello/'

def get_start():
    response = requests.post(url + 'joinSystem')
    _client_id = str(response.content.decode('utf-8'))
    print(int(_client_id))
    return int(_client_id)


def main(client_id):
    while True:
        response = requests.get(url + 'deepLearning')
        json_data = json.loads(response.text)
        # response_message = response.content().decode('utf-8')
        image_file_index = json_data['image_file_index']
        epoch_number = json_data['epoch_number']
        print('image_index_file: ' + str(image_file_index))
        print('epoch_number: ' + str(epoch_number))
        mode = str(json_data['mode'])
        if mode == 'done':
            return

        if mode == 'wait':
            time.sleep(1.5)
            continue

        l1_w_list = json_data['l1_w']
        l2_w_list = json_data['l2_w']
        lin_neural_network_l1 = L.Linear(784, 300)
        lin_neural_network_l2 = L.Linear(300, 10)
        for i in range(300):
            for j in range(784):
                lin_neural_network_l1.W.data[i][j] = l1_w_list[i][j]
        for i in range(10):
            for j in range(300):
                lin_neural_network_l2.W.data[i][j] = l2_w_list[i][j]
        mlp = MLP(lin_neural_network_l1, lin_neural_network_l2)
        file_images_name = '~/images_train/image_' + str(image_file_index)
        file_labels_name = '~/labels_train/label_' + str(image_file_index)
        if mode == "train":
            train(client_id, mlp, file_images_name, file_labels_name, l1_w_list, l2_w_list)
        if mode == "validation":
            validate(client_id, mlp, epoch_number, file_images_name, file_labels_name)


def train(client_id, mlp, file_images_name, file_labels_name, l1_w_list, l2_w_list):
    print('train')
    # Create a model
    model = L.Classifier(mlp)
    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    model.links()

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

    data = {
        'id': client_id,
        'mode': 'train',
        'work_time': total_time,
        'l1_delta': delta_layer1.tolist(),
        'l2_delta': delta_layer2.tolist()
    }
    print("total_time: " + str(total_time))
    requests.post(url + 'deepLearning"', json.dumps(data))


def validate(client_id, mlp, epoch_number, file_images_name, file_labels_name):
    # download validation files, images and labels.
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

    # fill the arrays of images the labels.
    label_array = np.fromstring(read_labels, dtype=np.int32, sep=',')
    images_array = [None] * 1200
    for i in range(0, 1200):
        images_array[i] = np.fromstring(read_image[i + 1], dtype=np.float32, sep=' ').reshape(1, 784)
    hit = 0
    for i in range(0, 1200):
        out = mlp.classify(images_array[i])
        if out == label_array[i]:
            hit += 1
    accuracy = (hit / 1200) * 100
    data = {
        'mode': 'validation' ,
        'accuracy': accuracy ,
        'epoch_number': epoch_number,
    }
    requests.post(url + 'deepLearning', json.dumps(data))
    return 0


if __name__ == '__main__':
    client_id = get_start()
    main(client_id)
