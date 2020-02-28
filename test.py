import torch

import torchvision.utils as utils

import os

import net
import data_loader
import utils as u

# device used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# use analytical AdaIN or MA module
use_MA_module = False


def compute_i_th_moment(input, i):
    """
    compute the i-th moment over all feature maps
    :param input:
    :param i:
    :return:
    """
    mean = torch.mean(input)
    std = torch.std(input)
    if i == 1:
        return mean
    elif i == 2:
        return std
    else:
        return torch.mean(((input - mean) / std).pow(i))


def test(configuration):
    """
    test loop
    :param configuration: the config file
    :return:
    """
    analytical_ada_in_module = net.AdaptiveInstanceNormalization()
    encoder = net.get_trained_encoder(configuration)
    decoder = net.get_trained_decoder(configuration)

    pretrained_model_path = configuration['pretrained_model_path']
    print('loading the moment alignment model from {}'.format(pretrained_model_path))

    content_images_path = configuration['content_images_path']
    style_images_path = configuration['style_images_path']

    loader = configuration['loader']
    unloader = configuration['unloader']

    image_saving_path = configuration['image_saving_path']
    moment_mode = configuration['moment_mode']

    moment_alignment_model = net.get_moment_alignment_model(configuration, moment_mode)
    print(moment_alignment_model)

    checkpoint = torch.load(pretrained_model_path, map_location=device)
    moment_alignment_model.load_state_dict(checkpoint)

    aligned_moment_loss = net.get_loss(configuration, moment_mode=moment_mode, lambda_1=0, lambda_2=10)

    number_content_images = len(os.listdir(content_images_path))
    number_style_images = len(os.listdir(style_images_path))

    content_image_files = ['{}/{}'.format(content_images_path, os.listdir(content_images_path)[i])
                           for i in range(number_content_images)]
    style_image_files = ['{}/{}'.format(style_images_path, os.listdir(style_images_path)[i])
                         for i in range(number_style_images)]

    for i in range(number_style_images):
        print("test_image {} at {}".format(i + 1, style_image_files[i]))

    for i in range(number_content_images):
        print("test_image {} at {}".format(i + 1, content_image_files[i]))

    for j in range(number_content_images):
        for i in range(number_style_images):
            style_image = data_loader.image_loader(style_image_files[i], loader)
            content_image = data_loader.image_loader(content_image_files[j], loader)
            with torch.no_grad():
                content_feature_maps = encoder(content_image)['r41']
                style_feature_maps = encoder(style_image)['r41']

                content_feature_map_batch_loader = data_loader.get_batch(content_feature_maps, 512)
                style_feature_map_batch_loader = data_loader.get_batch(style_feature_maps, 512)

                content_feature_map_batch = next(content_feature_map_batch_loader).to(device)
                style_feature_map_batch = next(style_feature_map_batch_loader).to(device)

                if use_MA_module:
                    style_feature_map_batch_moments = u.compute_moments_batches(style_feature_map_batch, last_moment=7)
                    content_feature_map_batch_moments = u.compute_moments_batches(content_feature_map_batch, last_moment=7)

                    out = moment_alignment_model(content_feature_map_batch,
                                                 content_feature_map_batch_moments,
                                                 style_feature_map_batch_moments,
                                                 is_test=True)

                    out_feature_map_batch_moments = u.compute_moments_batches(out, last_moment=7)

                    print_some_moments(style_feature_map_batch_moments, content_feature_map_batch_moments, out_feature_map_batch_moments)

                    loss, moment_loss, reconstruction_loss = aligned_moment_loss(content_feature_map_batch,
                                                                                 style_feature_map_batch,
                                                                                 content_feature_map_batch_moments,
                                                                                 style_feature_map_batch_moments,
                                                                                 out,
                                                                                 is_test=True)

                    print('loss: {}, moment_loss: {}, reconstruction_loss:{}'.format(
                        loss.item(), moment_loss.item(), reconstruction_loss.item()))
                else:
                    analytical_feature_maps = analytical_ada_in_module(content_feature_map_batch, style_feature_map_batch)
                    out = analytical_feature_maps

                utils.save_image([data_loader.imnorm(content_image, unloader),
                                  data_loader.imnorm(style_image, unloader),
                                  data_loader.imnorm(decoder(out.view(1, 512, 32, 32)), None)],
                                  '{}/A_image_{}_{}.jpeg'.format(image_saving_path, i,j), normalize=False)

                utils.save_image([data_loader.imnorm(decoder(out.view(1, 512, 32, 32)), None)],
                                 '{}/B_image_{}_{}.jpeg'.format(image_saving_path, i, j), normalize=False)


def print_some_moments(batch_1, batch_2, batch_3):
    """
    print some random moment of three batches
    :param batch_1: the first batch
    :param batch_2: the second batch
    :param batch_3: the third batch
    :return:
    """
    r = [int((torch.rand(1) * 512).item()), int((torch.rand(1) * 512).item()), int((torch.rand(1) * 512).item())]

    for i in range(3):
        print('style_pre_moments {}'.format(
            [batch_1[j].view(512)[r[i]].item() for j in range(len(batch_1))]
        ))
        print('content_pre_moments {}'.format(
            [batch_2[j].view(512)[r[i]].item() for j in range(len(batch_1))]
        ))
        print('result moments {}'.format(
            [batch_3[j].view(512)[r[i]].item() for j in range(len(batch_1))]
        ))
        print()

    print()
