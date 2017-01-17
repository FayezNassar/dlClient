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
from pymongo import MongoClient

url = 'https://deeplearningserver.herokuapp.com/hello/'


def get_start():
    response = requests.post(url + 'joinSystem')
    _client_id = str(response.content.decode('utf-8'))
    print(int(_client_id))
    return int(_client_id)


def main(client_id):
    while True:
        try:
            response = requests.get(url + 'deepLearning')
            json_data = json.loads(response.text)
            # response_message = response.content().decode('utf-8')
            image_file_index = int(json_data['image_file_index'])
            epoch_number = int(json_data['epoch_number'])
            print('image_index_file: ' + str(image_file_index))
            print('epoch_number: ' + str(epoch_number))
            mode = str(json_data['mode'])
            print('mode: ' + mode)
            if mode == 'stop':
                return

            if mode == 'wait':
                time.sleep(1.5)
                continue

            client = MongoClient('mongodb://Fayez:Fayez93@ds111529.mlab.com:11529/primre')
            _db = client.primre
            print('start download network')
            try:
                network = _db.Network.find_one({'id': 1})
                l1_w_list = network['l1_list']
                l2_w_list = network['l2_list']
            except:
                _db.GlobalParameters.update_one({'id': 1}, {'$inc': {'image_file_index': -1}})
                continue
            print('finish download network')
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
            if mode == 'test':
                file_images_name = '~/images_test/images_' + str(image_file_index)
                file_labels_name = '~/labels_test/label_' + str(image_file_index)

            if mode == "train":
                train(_db, client_id, mlp, file_images_name, file_labels_name, l1_w_list, l2_w_list)
            else:
                validate_test(_db, mode, mlp, epoch_number, file_images_name, file_labels_name)
        except:
            continue


def train(_db, client_id, mlp, file_images_name, file_labels_name, l1_w_list, l2_w_list):
    print('train')
    # Create a model
    model = L.Classifier(mlp)
    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    model.links()

    try:
        print('start download file')
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
        print('finish download file')
    except:
        _db.GlobalParameters.update_one({'id': 1}, {'$inc': {'image_file_index': -1}})
        return

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
            delta_layer2[i][j] = mlp.l2.W.data[i][j] - l2_w_list[i][j]
    for i in range(300):
        for j in range(784):
            delta_layer1[i][j] = mlp.l1.W.data[i][j] - l1_w_list[i][j]
    try:
        while _db.GlobalParameters.find_one({'id': 1})['list_busy'] == 1:
            continue
        _db.GlobalParameters.update({'id': 1}, {'$set': {'list_busy': 1}})
        network = _db.Network.find_one({'id': 1})
        l1_list = network['l1_list']
        l2_list = network['l2_list']
        for i in range(300):
            for j in range(784):
                l1_list[i][j] += (delta_layer1[i][j] * 0.1)  # 0.1 is the learning rate.
        for i in range(10):
            for j in range(300):
                l2_list[i][j] += (delta_layer2[i][j] * 0.1)  # 0.1 is the learning rate.
        _db.Network.update({'id': 1}, {'$set': {'l1_list': l1_list, 'l2_list': l2_list}})
        _db.GlobalParameters.update({'id': 1}, {'$set': {'list_busy': 0}})
    except:
        _db.GlobalParameters.update_one({'id': 1}, {'$inc': {'image_file_index': -1}})
        _db.GlobalParameters.update({'id': 1}, {'$set': {'list_busy': 0}})
        return
    data = {
        'id': client_id,
        'mode': 'train',
        'work_time': total_time,
    }
    print("total_time: " + str(total_time))
    requests.post(url + 'deepLearning"', json.dumps(data))


def validate_test(_db, mode, mlp, epoch_number, file_images_name, file_labels_name):
    print('validate_test, mode is: ' + str(mode))
    # download validation files, images and labels.
    try:
        print('start file download')
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
        print('finish file download')
    except:
        _db.GlobalParameters.update_one({'id': 1}, {'$inc': {'image_file_index': -1}})
        return

    # fill the arrays of images the labels.
    label_array = np.fromstring(read_labels, dtype=np.int32, sep=',')
    images_number = 1200 if (mode == 'train') else 200
    images_array = [None] * images_number
    for i in range(0, images_number):
        images_array[i] = np.fromstring(read_image[i + 1], dtype=np.float32, sep=' ').reshape(1, 784)
    hit = 0
    for i in range(0, images_number):
        out = mlp.classify(images_array[i])
        if out == label_array[i]:
            hit += 1
    accuracy = (hit / images_number) * 100
    data = {
        'mode': mode,
        'accuracy': accuracy,
        'epoch_number': epoch_number,
    }
    requests.post(url + 'deepLearning', json.dumps(data))
    return 0


if __name__ == '__main__':
    client_id = get_start()
    main(client_id)
