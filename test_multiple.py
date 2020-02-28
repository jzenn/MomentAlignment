import torch
import torchvision.utils as u

import os

import net as net
import utils
import data_loader as data_loader

# device used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(configuration):
    """
    test loop to produce images with multiple models
    :param configuration:
    :return:
    """
    encoder = net.get_trained_encoder(configuration)
    decoder = net.get_trained_decoder(configuration)

    model_path_list = configuration['model_path_list']
    content_images_path = configuration['content_images_path']
    style_images_path = configuration['style_images_path']
    loader = configuration['loader']
    image_saving_path = configuration['image_saving_path']
    mode_list = configuration['mode_list']

    model_list = [0 for _ in range(len(model_path_list))]
    for i in range(len(model_list)):
        moment_alignment_model = net.get_moment_alignment_model(configuration, moment_mode=mode_list[i], use_list=True, list_index=i)
        checkpoint = torch.load(model_path_list[i], map_location='cpu')
        moment_alignment_model.load_state_dict(checkpoint)
        model_list[i] = moment_alignment_model

    number_content_images = len(os.listdir(content_images_path))
    number_style_images = len(os.listdir(style_images_path))
    content_image_files = ['{}/{}'.format(content_images_path, sorted(os.listdir(content_images_path))[i])
                           for i in range(number_content_images)]
    style_image_files = ['{}/{}'.format(style_images_path, sorted(os.listdir(style_images_path))[i])
                         for i in range(number_style_images)]

    for i in range(number_content_images):
        for j in range(number_style_images):
            print('at image {}'.format(i))
            with torch.no_grad():
                content_image = data_loader.image_loader(content_image_files[i], loader)
                style_image = data_loader.image_loader(style_image_files[j], loader)

                result_images = [0 for _ in range(len(model_list))]
                for k in range(len(model_list)):
                    content_feature_map_batch_loader = data_loader.get_batch(encoder(content_image)['r41'], 512)
                    style_feature_map_batch_loader = data_loader.get_batch(encoder(style_image)['r41'], 512)
                    content_feature_map_batch = next(content_feature_map_batch_loader).to(device)
                    style_feature_map_batch = next(style_feature_map_batch_loader).to(device)

                    style_feature_map_batch_moments = utils.compute_moments_batches(style_feature_map_batch,
                                                                                    last_moment=7)
                    content_feature_map_batch_moments = utils.compute_moments_batches(content_feature_map_batch,
                                                                                      last_moment=7)

                    result_images[k] = decoder(model_list[k](content_feature_map_batch, content_feature_map_batch_moments,
                                                             style_feature_map_batch_moments).view(1, 512, 32, 32))
                    result_images[k] = result_images[k].squeeze(0)

                # save all images in one row
                u.save_image([data_loader.imnorm(content_image, None), data_loader.imnorm(style_image, None)]
                             + result_images,
                             '{}/moment_alignment_test_image_A_{}_{}.jpeg'.format(image_saving_path, i, j),
                             normalize=False, scale_each=False, pad_value=1)

                # save all images in two rows
                u.save_image([data_loader.imnorm(content_image, None), data_loader.imnorm(style_image, None)]
                             + [torch.ones(3, 256, 256) for _ in range(2)]
                             + result_images,
                             '{}/moment_alignment_test_image_B_{}_{}.jpeg'.format(image_saving_path, i, j),
                             normalize=False, scale_each=False, pad_value=1, nrow=4)

                # save all result images in one row
                u.save_image(result_images,
                             '{}/moment_alignment_test_image_C_{}_{}.jpeg'.format(image_saving_path, i, j),
                             normalize=False, scale_each=False, pad_value=1)

                # save all result images in one row + content image
                u.save_image([data_loader.imnorm(content_image, None)] + result_images,
                             '{}/moment_alignment_test_image_D_{}_{}.jpeg'.format(image_saving_path, i, j),
                             normalize=False, scale_each=False, pad_value=1)

                # save all result images in one row + style image
                u.save_image([data_loader.imnorm(style_image, None)] + result_images,
                             '{}/moment_alignment_test_image_E_{}_{}.jpeg'.format(image_saving_path, i, j),
                             normalize=False, scale_each=False, pad_value=1)
