import torch

import torchvision.utils as utils

import os

import utils as u
import net
import data_loader

# device used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_n_th_moment(x, n):
    mean = torch.mean(x)
    std = torch.std(x)
    if n == 1:
        return mean
    elif n == 2:
        return std
    else:
        return torch.mean(((x - mean) / std).pow(n))


def test(configuration):
    """
    test the moment alignment solution (for 2 moments) in comparison
    to the analytical solution that can be computed for mean and std
    :param configuration: the config file
    :return:
    """
    encoder = net.get_trained_encoder(configuration)
    decoder = net.get_trained_decoder(configuration)

    analytical_ada_in_module = net.AdaptiveInstanceNormalization()

    pretrained_model_path = configuration['pretrained_model_path']
    content_images_path = configuration['content_images_path']
    style_images_path = configuration['style_images_path']
    loader = configuration['loader']
    unloader = configuration['unloader']
    image_test_saving_path = configuration['image_saving_path']
    moment_mode = configuration['moment_mode']

    print('loading the moment alignment model from {}'.format(pretrained_model_path))
    moment_alignment_model = net.get_moment_alignment_model(configuration, moment_mode)
    print(moment_alignment_model)

    checkpoint = torch.load(pretrained_model_path, map_location=device)
    moment_alignment_model.load_state_dict(checkpoint)

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

    iterations = 0
    mean_percentages = [0, 0, 0, 0, 0, 0, 0]

    for j in range(number_content_images):
        for i in range(number_style_images):
            style_image = data_loader.image_loader(style_image_files[i], loader)
            content_image = data_loader.image_loader(content_image_files[j], loader)
            with torch.no_grad():
                content_feature_maps = encoder(content_image)['r41']
                style_feature_maps = encoder(style_image)['r41']

                content_feature_map_batch_loader = data_loader.get_batch(content_feature_maps,
                                                                         512)
                style_feature_map_batch_loader = data_loader.get_batch(style_feature_maps,
                                                                       512)

                content_feature_map_batch = next(content_feature_map_batch_loader).to(device)
                style_feature_map_batch = next(style_feature_map_batch_loader).to(device)

                style_feature_map_batch_moments = u.compute_moments_batches(style_feature_map_batch)
                content_feature_map_batch_moments = u.compute_moments_batches(content_feature_map_batch)

                out = moment_alignment_model(content_feature_map_batch,
                                             content_feature_map_batch_moments,
                                             style_feature_map_batch_moments)

                result_feature_maps = out

                analytical_feature_maps = analytical_ada_in_module(content_feature_map_batch, style_feature_map_batch)

                a_0, a_001, a_001_l, a_01, a_01_l, a_1, a_1_l = \
                    get_distance(analytical_feature_maps, result_feature_maps)
                iterations += 1

                mean_percentages[0] += a_0
                mean_percentages[1] += a_001
                mean_percentages[2] += a_001_l
                mean_percentages[3] += a_01
                mean_percentages[4] += a_01_l
                mean_percentages[5] += a_1
                mean_percentages[6] += a_1_l

                # u.imshow(decoder(analytical_feature_maps.view(1, 512, 32, 32)), transforms.ToPILImage())

                utils.save_image([data_loader.imnorm(content_image, unloader),
                                  data_loader.imnorm(style_image, unloader),
                                  data_loader.imnorm(decoder(result_feature_maps.view(1, 512, 32, 32)), None),
                                  data_loader.imnorm(decoder(analytical_feature_maps.view(1, 512, 32, 32)), None)],
                                  '{}/A_image_{}_{}.jpeg'.format(image_test_saving_path, i, j), normalize=False, pad_value=1)

                utils.save_image([data_loader.imnorm(content_image, unloader),
                                  data_loader.imnorm(decoder(result_feature_maps.view(1, 512, 32, 32)), None),
                                  data_loader.imnorm(decoder(analytical_feature_maps.view(1, 512, 32, 32)),
                                                     None)],
                                 '{}/B_image_{}_{}.jpeg'.format(image_test_saving_path, i, j), normalize=False, pad_value=1)

                utils.save_image([data_loader.imnorm(style_image, unloader),
                                  data_loader.imnorm(decoder(result_feature_maps.view(1, 512, 32, 32)), None),
                                  data_loader.imnorm(decoder(analytical_feature_maps.view(1, 512, 32, 32)),
                                                     None)],
                                 '{}/C_image_{}_{}.jpeg'.format(image_test_saving_path, i, j), normalize=False, pad_value=1)

                utils.save_image([data_loader.imnorm(decoder(result_feature_maps.view(1, 512, 32, 32)), None),
                                  data_loader.imnorm(decoder(analytical_feature_maps.view(1, 512, 32, 32)),
                                                     None)],
                                 '{}/D_image_{}_{}.jpeg'.format(image_test_saving_path, i, j), normalize=False, pad_value=1)

    print('averaging percentages')
    mean_percentages = [mean_percentages[i]/iterations for i in range(len(mean_percentages))]
    print(mean_percentages)


def get_distance(analytical_feature_maps, result_feature_maps):
    analytical_feature_maps = analytical_feature_maps.view(1, 512, 32, 32)
    result_feature_maps = result_feature_maps.view(1, 512, 32, 32)

    eps = 1e-6

    res_alpha_0 = torch.mean(
        torch.abs(analytical_feature_maps - result_feature_maps) / (torch.abs(analytical_feature_maps) + eps)) * 100

    res_alpha_001 = torch.mean(
        torch.abs(analytical_feature_maps[torch.abs(analytical_feature_maps) > 0.01]
                  - result_feature_maps[torch.abs(analytical_feature_maps) > 0.01])
        / (torch.abs(analytical_feature_maps[torch.abs(analytical_feature_maps) > 0.01]) + eps)) * 100

    res_alpha_001_l = torch.mean(
        torch.abs(analytical_feature_maps[torch.abs(analytical_feature_maps) <= 0.01]
                  - result_feature_maps[torch.abs(analytical_feature_maps) <= 0.01])
        / (torch.abs(analytical_feature_maps[torch.abs(analytical_feature_maps) <= 0.01]) + eps)) * 100

    res_alpha_01 = torch.mean(
        torch.abs(analytical_feature_maps[torch.abs(analytical_feature_maps) > 0.1]
                  - result_feature_maps[torch.abs(analytical_feature_maps) > 0.1])
        / (torch.abs(analytical_feature_maps[torch.abs(analytical_feature_maps) > 0.1]) + eps)) * 100

    res_alpha_01_l = torch.mean(
        torch.abs(analytical_feature_maps[torch.abs(analytical_feature_maps) <= 0.1]
                  - result_feature_maps[torch.abs(analytical_feature_maps) <= 0.1])
        / (torch.abs(analytical_feature_maps[torch.abs(analytical_feature_maps) <= 0.1]) + eps)) * 100

    res_alpha_1 = torch.mean(
        torch.abs(analytical_feature_maps[torch.abs(analytical_feature_maps) > 1]
                  - result_feature_maps[torch.abs(analytical_feature_maps) > 1])
        / (torch.abs(analytical_feature_maps[torch.abs(analytical_feature_maps) > 1]) + eps)) * 100

    res_alpha_1_l = torch.mean(
        torch.abs(analytical_feature_maps[torch.abs(analytical_feature_maps) <= 1]
                  - result_feature_maps[torch.abs(analytical_feature_maps) <= 1])
        / (torch.abs(analytical_feature_maps[torch.abs(analytical_feature_maps) <= 1]) + eps)) * 100

    print('averaged per-pixel deviation (a = 0   ,  ): {}'.format(res_alpha_0))
    print('averaged per-pixel deviation (a = 0.01, >): {}'.format(res_alpha_001))
    print('averaged per-pixel deviation (a = 0.01, <): {}'.format(res_alpha_001_l))
    print('averaged per-pixel deviation (a = 0.1 , >): {}'.format(res_alpha_01))
    print('averaged per-pixel deviation (a = 0.1 , <): {}'.format(res_alpha_01_l))
    print('averaged per-pixel deviation (a = 1   , >): {}'.format(res_alpha_1))
    print('averaged per-pixel deviation (a = 1   , <): {}'.format(res_alpha_1_l))
    print()

    return res_alpha_0, res_alpha_001, res_alpha_001_l, res_alpha_01, res_alpha_01_l, res_alpha_1, res_alpha_1_l
