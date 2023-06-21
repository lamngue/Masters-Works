# This is a file where you should put your own functions
import torch
import sys
import torch.nn as nn
import torchvision
import tensorflow as tf
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn.utils.prune as prune
import csv

from torchvision import datasets, transforms
from torch.nn.utils.prune import l1_unstructured, random_unstructured
from d2l import torch as d2l
device = d2l.try_gpu()

WRITE_TO_FILE = True
NOT_WRITE_TO_FILE = False
SIMPLE_TRAINING = True
NOT_SIMPLE_TRAINING = False
BEST_MODEL_SO_FAR = True
NOT_BEST_MODEL_SO_FAR = False
ORIGINAL_MODEL = True
NOT_ORIGINAL_MODEL = False
RANDOM_INIT = True
NOT_RANDOM_INIT = False

# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------

# TODO: Datasets go here.


def load_datasets(dataset_name):

    if dataset_name == 'mnist':
        data_train_large = datasets.MNIST(
            root='data', train=True, transform=transforms.ToTensor(), download=True)
        data_test = datasets.MNIST(
            root='data', train=False, transform=transforms.ToTensor(), download=True)
        n_train, n_val = 55000, 5000
    elif dataset_name == 'cifar10':
        data_train_large = datasets.CIFAR10(
            root='data', train=True, transform=transforms.ToTensor(), download=True)
        data_test = datasets.CIFAR10(
            root='data', train=False, transform=transforms.ToTensor(), download=True)
        n_train, n_val = 45000, 5000
    else:
        raise Exception(f"Unknown dataset: {dataset_name}")
    # perform train/val split
    data_train, data_val = torch.utils.data.random_split(
        data_train_large, [n_train, n_val])
    
    
    return data_train, data_val, data_test


# -----------------------------------------------------------------------------
# Network architectures
# -----------------------------------------------------------------------------

# TODO: Define network architectures here
# VGG19
vgg19 = models.vgg19()

# Resnet18
resnet18 = models.resnet18()


# Conv-2
class Conv2(nn.Module):
    def __init__(self):
        super(Conv2, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(16384, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.ReLU(),
            nn.Softmax(),
        )

    def forward(self, x):
        return self.net(x)


# Conv-4
class Conv4(nn.Module):
    def __init__(self):
        super(Conv4, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(8192, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.ReLU(),
            nn.Softmax(),
        )

    def forward(self, x):
        return self.net(x)


# Conv-6
class Conv6(nn.Module):
    def __init__(self):
        super(Conv6, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.ReLU(),
            nn.Softmax(),
        )

    def forward(self, x):
        return self.net(x)


# Lenet-300-100
class LeNet_300_100(nn.Module):
    def __init__(self):
        super(LeNet_300_100, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        return self.net(x)


def create_network(arch, **kwargs):
    # TODO: Change this function for the architectures you want to support
    if arch == 'lenet':
        return LeNet_300_100(**kwargs)
    elif arch == 'conv-2':
        return Conv2(**kwargs)
    elif arch == 'conv-4':
        return Conv4(**kwargs)
    elif arch == 'conv-6':
        return Conv6(**kwargs)
    elif arch == 'Resnet18':
        return resnet18(**kwargs)
    elif arch == 'Vgg19':
        return vgg19(**kwargs)
    else:
        raise Exception(f"Unknown architecture: {arch}")

# -----------------------------------------------------------------------------
# Training and testing loops
# -----------------------------------------------------------------------------


def evaluate(net, dataset, device=d2l.try_gpu()):
    # todo maybe d2l.evaluate_loss and d2l.evaluate_accuracy_gpu?
    loss = nn.CrossEntropyLoss()
    metric = d2l.Accumulator(3)
    for i, (X, y) in enumerate(dataset):
        with torch.no_grad():
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
    eval_loss = metric[0] / metric[2]
    eval_acc = metric[1] / metric[2]
    return eval_loss, eval_acc


def save_model(model, arch, cur_run, phase, is_simple, is_best, iteration):
    if not is_simple:
        if not is_best:
            torch.save(model.state_dict(), f'checkpoints/{arch}-{cur_run}-{phase}-{iteration}-start.pth')
        else:
            torch.save(model.state_dict(), f'checkpoints/{arch}-{cur_run}-{phase}-{iteration}-best.pth')
    else:
        torch.save(model.state_dict(),
                   f'checkpoints_simple/{arch}-{cur_run}-{phase}-.pth')



def load_model(arch, run, phase, is_simple, is_best, iteration):
    if not is_simple:
        if not is_best:
            return torch.load(f'checkpoints/{arch}-{run}-{phase}-{iteration}-start.pth')
        else:
            return torch.load(f'checkpoints/{arch}-{run}-{phase}-{iteration}-best.pth')
    else:
        return torch.load(f'checkpoints_simple/{arch}-{run}-{phase}.pth')



def get_params(arch):
    # see Figure 2 of the paper
    if arch == 'lenet':
        return {'n_iterations': 50000, 'batch_size': 60, 'optimizer': torch.optim.Adam, 'lr': 1.2e-3,
                'pruning_rate': {'fully_connected': .2}}
    if arch == 'conv-2':
        return {'n_iterations': 20000, 'batch_size': 60, 'optimizer': torch.optim.Adam, 'lr': 2e-4,
                'pruning_rate': {'conv': .1, 'fully_connected': .2}}
    elif arch == 'conv-4':
        return {'n_iterations': 25000, 'batch_size': 60, 'optimizer': torch.optim.Adam, 'lr': 3e-4,
                'pruning_rate': {'conv': .1, 'fully_connected': .2}}
    elif arch == 'conv-6':
        return {'n_iterations': 30000, 'batch_size': 60, 'optimizer': torch.optim.Adam, 'lr': 3e-4,
                'pruning_rate': {'conv': .15, 'fully_connected': .2}}
    elif arch == 'Resnet18':
        return {'n_iterations': 30000, 'batch_size': 128, 'optimizer': torch.optim.SGD, 'lr': .1 - .01 - .001,
                'pruning_rate': {'conv': .2, 'fully_connected': .0}}
    elif arch == 'Vgg19':
        return {'n_iterations': 112000, 'batch_size': 64, 'optimizer': torch.optim.SGD, 'lr': .9,  # todo this is wrong???
                'pruning_rate': {'conv': .2, 'fully_connected': .0}}
    else:
        raise Exception(f"Unknown architecture: {arch}")


def train(net, arch, params, dataset_name, device=d2l.try_gpu()):

    n_iterations = params['n_iterations']
    batch_size = params['batch_size']

    optimizer = params['optimizer'](net.parameters(), lr=params['lr'])

    train_iter, val_iter, test_iter = load_datasets(
        dataset_name)
    iters_per_epoch = len(train_iter) / batch_size
    epochs = int(n_iterations / iters_per_epoch)
    

    train_iter = torch.utils.data.DataLoader(
        train_iter, batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(
        val_iter, batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(
        test_iter, batch_size, shuffle=True)

    loss = nn.CrossEntropyLoss()

   
    animator = d2l.Animator(xlabel='epoch',
                                legend=['train loss', 'train acc',
                                        'validation acc'],
                                figsize=(10, 5))

    timer, num_batches = d2l.Timer(), len(train_iter)

    cur_run = 1
    for epoch in range(epochs):
        metric = d2l.Accumulator(3)
        cur_run += 1
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if ((i + 1) % (num_batches // 5) == 0 or i == num_batches - 1):
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))
        test_loss, test_acc = evaluate(net, test_iter)
        train_loss, train_acc = evaluate(net, train_iter)

  
        animator.add(epoch + 1, (None, None, test_acc))

    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * n_iterations / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

# TODO: Define training, testing and model loading here

# -----------------------------------------------------------------------------
# Pruning
# -----------------------------------------------------------------------------

# TODO: Put functions related to pruning here


def check_pruned_linear(linear):
    params = {param_name for param_name, _ in linear.named_parameters()}
    expected_params = {"weight_orig", "bias_orig"}

    return params == expected_params


def prune_network_linear(amount, module, method='l1'):
    if method == 'l1':
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            l1_unstructured(module, 'weight', amount)
    else:
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            random_unstructured(module, 'weight', amount)


def prune_network_sequential(amount, module, method='l1'):
    if isinstance(amount, float):
        amounts = [amount] * len(module)

    for amount, layer in zip(amounts, module):
        prune_network_linear(amount, layer, method=method)


def copy_pruning_mask(unpruned, pruned):
    for pruned_layer, unpruned_layer in zip(pruned, unpruned):
        with torch.no_grad():
            if (isinstance(pruned_layer, nn.Linear) or isinstance(pruned_layer, nn.Conv2d)):
                prune.custom_from_mask(unpruned_layer, 'weight', pruned_layer.weight_mask)            



def copy_weights_linear(linear_unpruned, linear_pruned):
    if isinstance(linear_pruned, nn.Linear) or isinstance(linear_pruned, nn.Conv2d):
        with torch.no_grad():
            linear_pruned.weight_orig.copy_(linear_unpruned.weight)
            linear_pruned.bias_orig.copy_(linear_unpruned.bias)


def copy_weights_sequential(unpruned, pruned):
    for linear_unpruned, linear_pruned in zip(unpruned, pruned):
        copy_weights_linear(linear_unpruned, linear_pruned)


# Reset the remaining parameters in the pruned to the original theta_0
def apply_mask_linear(linear_original, linear_pruned):
    if isinstance(linear_original, nn.Linear) or isinstance(linear_original, nn.Conv2d):
        with torch.no_grad():
            new_weight = linear_original.weight.mul(linear_pruned.weight_mask)
            linear_pruned.weight.copy_(new_weight)


def apply_mask_sequential(original_net, pruned_net):
    for linear_original, linear_pruned in zip(original_net, pruned_net):
        apply_mask_linear(linear_original, linear_pruned)



def get_network_and_params(arch):
    params = get_params(arch)
    net = create_network(arch)
    net = net.to(device)
    return net, params


def train_with_early_stop_calculations(net, arch, cur_run, amount, params, dataset_name, file_name, write_to_file, original_model, iteration = 1, fig_num = 1):

    n_iterations = params['n_iterations']
    batch_size = params['batch_size']

    optimizer = params['optimizer'](net.parameters(), lr=params['lr'])

    train_iter, val_iter, test_iter = load_datasets(
        dataset_name)
    iters_per_epoch = len(train_iter) / batch_size
    epochs = int(n_iterations / iters_per_epoch)
    timer, num_batches = d2l.Timer(), len(train_iter)


    train_iter = torch.utils.data.DataLoader(
        train_iter, batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(
        val_iter, batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(
        test_iter, batch_size, shuffle=True)


    loss = nn.CrossEntropyLoss()

    temp_val_loss = sys.maxsize
    threshold = 150
    current = 0
    iter = 1

    with open(file_name, 'a+') as f:
        for epoch in range(epochs):
            if iter >= 20000 and fig_num == 3:
                break
            metric = d2l.Accumulator(3)
            net.train()
            for i, (X, y) in enumerate(train_iter):
                timer.start()
                optimizer.zero_grad()
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                l = loss(y_hat, y)
                l.backward()
                optimizer.step()
                with torch.no_grad():
                    metric.add(l * X.shape[0],
                               d2l.accuracy(y_hat, y), X.shape[0])
                timer.stop()
                train_l = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                iter += 1
                current += 1
                if (current == threshold):  # threshold = 150
                    current = 0
                    val_loss, _ = evaluate(net, val_iter)
                    if val_loss < temp_val_loss:
                        temp_val_loss = val_loss

                        if original_model:
                            save_model(net, arch, cur_run, 'running-original', NOT_SIMPLE_TRAINING, BEST_MODEL_SO_FAR, iteration)
                        else:
                            save_model(net, arch, cur_run, 'running-winning-ticket', NOT_SIMPLE_TRAINING, BEST_MODEL_SO_FAR, iteration)

                        if write_to_file and fig_num != 3:
                            remaining_percentage = 1 - amount
                            f.write(str(remaining_percentage) + "," + str(iter) + "\n")
                if iter % 5000 == 0 and fig_num == 3 and not original_model:
                    _, test_acc = evaluate(net, test_iter)
                    f.write(str(test_acc) + "," + str(iter) + "\n")
                    f.write("------------------\n")
            _, test_acc = evaluate(net, test_iter)
        if fig_num == 4 and write_to_file:
            f.write("End of training for model")
    f.close()

    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * n_iterations / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


def calculate_stats_and_write_to_file(best_model,dataset_name, params, file_name, fig_num, weight_remaining):
    train_iter, val_iter, test_iter = load_datasets(
        dataset_name)

    val_iter = torch.utils.data.DataLoader(
        val_iter, params['batch_size'], shuffle=True)


    train_iter = torch.utils.data.DataLoader(
        train_iter, params['batch_size'], shuffle=True)
    test_iter = torch.utils.data.DataLoader(
        test_iter, params['batch_size'], shuffle=True)

    _, val_acc = evaluate(best_model, val_iter)

    _, test_acc = evaluate(best_model, test_iter)
    _, train_acc = evaluate(best_model, train_iter)

    with open(file_name, 'a+') as f:
        f.write(
               "Writing the remaining amount and validation accuracy for figure " + str(fig_num) + "\n")
        if fig_num == 1:
            f.write(str(weight_remaining) + "," + str(val_acc) + "\n")
        elif fig_num == 4:
            f.write(str(weight_remaining) + "," +
                        str(test_acc) + "," + str(train_acc) + "\n")
        f.close()


# One-shot
def section1_oneshot(dataset_name, arch, cur_run, amount, method, file_name, fig_num = 1, random_init = False):

    # original_model
    net, params = get_network_and_params(arch)
    save_model(net, arch, cur_run, 'init', NOT_SIMPLE_TRAINING, NOT_BEST_MODEL_SO_FAR, 1)
    

    train_with_early_stop_calculations(net, arch, cur_run, amount, params, dataset_name, file_name, NOT_WRITE_TO_FILE, ORIGINAL_MODEL)

    best_model, _ = get_network_and_params(arch)
    best_model.load_state_dict(load_model(arch, cur_run, 'running-original', NOT_SIMPLE_TRAINING, BEST_MODEL_SO_FAR,1))
    prune_network_sequential(amount, best_model.net, method)

    if fig_num == 1:
        net_copy, _ = get_network_and_params(arch)
        net_copy.load_state_dict(load_model(arch, cur_run, 'init', NOT_SIMPLE_TRAINING, NOT_BEST_MODEL_SO_FAR,1))

        # Reset the remaining parameters from best_model to their values in original model
        apply_mask_sequential(net_copy.net, best_model.net)
    if random_init: # random reinit 3 times
        for _ in range(3):
            net_copy, _ = get_network_and_params(arch)
            apply_mask_sequential(net_copy.net, best_model.net)

    train_with_early_stop_calculations(best_model, arch, cur_run, amount, params, dataset_name, file_name, WRITE_TO_FILE, NOT_ORIGINAL_MODEL)

    best_model.load_state_dict(load_model(
        arch, cur_run, 'running-winning-ticket', NOT_SIMPLE_TRAINING, BEST_MODEL_SO_FAR, 1))
   
    if method == 'random':
        file_name = arch + '/train_results_iteration_'+ arch +'_random.txt'
    calculate_stats_and_write_to_file(best_model, dataset_name, params, file_name, fig_num, 1-amount)

    return best_model

# Iterative
def section2_iterative(dataset_name, arch, cur_run, amount, n, method, file_name, fig_num = 3, random_init = False):
    # original_model
    net, params = get_network_and_params(arch)
    save_model(net, arch, cur_run, 'init', NOT_SIMPLE_TRAINING,
               NOT_BEST_MODEL_SO_FAR, 1)
    per_round_prune_ratio = 1 - (1 - amount) ** (1 / n)

    best_model, _ = get_network_and_params(arch)
    weight_remaining = 1

    net_copy, _ = get_network_and_params(arch)

    for i in range(n):
        weight_remaining -= per_round_prune_ratio

        train_with_early_stop_calculations(
            best_model, arch, cur_run, amount, params, dataset_name, file_name, NOT_WRITE_TO_FILE, ORIGINAL_MODEL, i+1, 3)
        best_model.load_state_dict(load_model(
            arch, cur_run, 'running-original', NOT_SIMPLE_TRAINING, BEST_MODEL_SO_FAR, i + 1))
        prune_network_sequential(
            per_round_prune_ratio, best_model.net, method)
        
        save_model(best_model, arch, cur_run, 'running-original', NOT_SIMPLE_TRAINING, BEST_MODEL_SO_FAR, i+1)
        net_copy.load_state_dict(load_model(arch, cur_run, 'init', NOT_SIMPLE_TRAINING, NOT_BEST_MODEL_SO_FAR, 1))
        # Reset the remaining parameters from best_model to their values in original model
        apply_mask_sequential(net_copy.net, best_model.net)
    
    # Train the final network
    if random_init:  # random reinit 3 times
        for _ in range(3):
            net_copy, _ = get_network_and_params(arch)
            apply_mask_sequential(net_copy.net, best_model.net)

    train_with_early_stop_calculations(
            best_model, arch, cur_run, amount, params, dataset_name, file_name, WRITE_TO_FILE, NOT_ORIGINAL_MODEL, n, 3)
    best_model.load_state_dict(load_model(
        arch, cur_run, 'running-winning-ticket', NOT_SIMPLE_TRAINING, BEST_MODEL_SO_FAR, n))
    if fig_num == 4:
        calculate_stats_and_write_to_file(best_model,dataset_name,params,file_name,4,weight_remaining)

    return (weight_remaining, best_model)

# figure 4 compares one-shot and iterative pruning
def figure4(dataset_name, arch, cur_run, amount, n, file_name):
    # Winning ticket - one shot
    # original_model

    section1_oneshot(dataset_name, arch, cur_run, amount, "l1", file_name, 4, NOT_RANDOM_INIT)

    # Random Reinit 3 times - one shot
    section1_oneshot(dataset_name, arch, cur_run, amount, 'l1', file_name, 4, RANDOM_INIT)
   

    # Winning ticket - iterative
    section2_iterative(dataset_name, arch, cur_run,
                       1-(1-amount/n)**n, n, 'l1', file_name, 4, NOT_RANDOM_INIT)

    # Random Reinit 3 times - iterative
    section2_iterative(dataset_name, arch, cur_run,
                       1-(1-amount/n)**n, n, 'l1', file_name, 4, RANDOM_INIT)
    

    
    
