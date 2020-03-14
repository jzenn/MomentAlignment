import torch

import matplotlib.pyplot as plt

import datetime
import pytz
import os
import glob

# the date-time format
fmt = '%d_%m__%H_%M_%S'

plt.ion()


def imshow(tensor, transformation, title=None):
    """
    shows an image in a plot
    :param tensor: the image as tensor
    :param transformation: the transformation applied to the image
    :param title: the title of the plot
    :return:
    """
    # clone the tensor to not change the original one
    image = tensor.cpu().clone()
    # remove the batch dimension
    image = image.squeeze(0)
    if transformation is not None:
        image = transformation(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    # pause that the plot can get updated
    plt.pause(0.005)


def compute_moments_batches(input, last_moment=8):
    """
    compute the moments in batches for every batch and feature map
    :param input: tensor
    :param last_moment: the last moment to be computed on the batch
    :return: properly sized tensor with the moments
    """
    out = []
    for i in range(1, last_moment + 1):
        out += [compute_i_th_moment_batches(input, i)]

    return out


def compute_i_th_moment_batches(input, i):
    """
    compute the i-th moment for every feature map in the batch
    :param input: tensor
    :param i: the moment to be computed
    :return:
    """
    n, c, h, w = input.size()
    input = input.view(n, c, -1)
    mean = torch.mean(input, dim=2).view(n, c, 1, 1)
    eps = 1e-5
    var = torch.var(input, dim=2).view(n, c, 1, 1) + eps
    std = torch.sqrt(var)
    if i == 1:
        return mean
    elif i == 2:
        return std
    else:
        sol = ((input.view(n, c, h, w) - mean.expand(n, c, h, w)) / std).pow(i)
        sol = torch.mean(sol.view(n, c, -1), dim=2).view(n, c, 1, 1)
        return sol


def compute_moments(input):
    """
    compute the moments for a batch of size 1
    :param input: tensor
    :return:
    """
    out = []
    for i in range(1, 9):
        out += [compute_i_th_moment(input, i)]

    return out


def compute_i_th_moment(input, n):
    """
    compute the i-th moment for every feature map in the batch of size 1
    :param input: tensor
    :param i: the moment to be computed
    :return:
    """
    mean = torch.mean(input)
    eps = 1e-5
    var = torch.var(input) + eps
    std = torch.sqrt(var)
    if n == 1:
        return mean
    elif n == 2:
        return std
    else:
        res = torch.mean(((input - mean) / std).pow(n))
        return res


def calc_mean_and_std(input):
    """
    calculates mean and std channel-wise (R,G,B)
    :param input:
    :return: mean an std (channel-wise)
    """
    assert (len(input.size()) == 4), "the size of the feature map should not be {}".format(input.size())

    input_size = input.size()
    # (n, c, h, w)
    # n = #batch, c = #channels, h = height, w = width
    n = input_size[0]
    c = input_size[1]

    mean = torch.mean(input.view(n, c, -1), dim=2, keepdim=True).view(n, c, 1, 1)

    # prevent division by zero with eps
    eps = 1e-5
    var = torch.var(input.view(n, c, -1), dim=2) + eps
    std = torch.sqrt(var).view(n, c, 1, 1)

    # std = torch.std(input.view(n,c,-1), dim=2, keepdim=True).view(n,c,1,1)

    return mean, std


def save_current_model(lambda_1, lambda_2, model_state_dict, optimizer_state_dict, model_saving_path):
    """
    save the current models state_dict as well as the optimizers state_dict
    :param lambda_1: reconstruction loss
    :param lambda_2: moment loss
    :param model_state_dict: the model state dict
    :param optimizer_state_dict: the optimizer state dict
    :param model_saving_path: the folder where the models should be saved to
    :return:
    """
    torch.save({
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            }, '{}/moment_alignment_model__{}_{}__{}.pt'.format(model_saving_path, lambda_1, lambda_2,
                                                             datetime.datetime.now(pytz.utc).astimezone(
                                                                 pytz.timezone('Europe/Berlin')).strftime(fmt)))


def save_current_best_model(epoch, model, model_saving_path):
    """
    saves the current best model (every validation interval)
    :param epoch: the current epoch
    :param model: the current model
    :param model_saving_path: the path where the model is saved to
    :return:
    """
    try:
        model_state_dict = model.module.state_dict()
    except:
        model_state_dict = model.state_dict()

    torch.save(model_state_dict, '{}/moment_alignment_model__{}.pt'.format(model_saving_path, epoch))


def get_latest_model(configuration):
    """
    get the latest model from model_path
    :param configuration:
    :return:
    """
    model_path = configuration['model_path'] + '/*'
    latest_file = max(glob.iglob(model_path), key=os.path.getctime)
    return latest_file
