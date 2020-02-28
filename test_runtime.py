import torch
import torchvision.transforms as transforms
import torchvision.utils as u

import os

import utils
import data_loader
import time
import net
import numpy as np

# local dev
local = True

# device used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

prefix = '/Users/Johannes/Desktop/mom_al_deep/' if local else '/home/zenn/bachelor_thesis/pytorch_models/'
suffix = ''
model_path_list = [
    prefix + 'moment_alignment_model_0_1_2_0' + suffix,
    prefix + 'moment_alignment_model_0_1_3_0' + suffix,
    prefix + 'moment_alignment_model_0_1_4_0' + suffix,
    prefix + 'moment_alignment_model_0_1_5_0' + suffix,
]

model_configuration = {
    'do_incremental_training': False,
    'use_very_shallow_model_list': [False for i in range(len(model_path_list))],
    'load_model': False
}

model_list = [None for _ in range(len(model_path_list))]
for i in range(len(model_list)):
    moment_alignment_model = net.get_moment_alignment_model(model_configuration, moment_mode=i + 2, use_list=True, list_index=i)
    checkpoint = torch.load(model_path_list[i], map_location='cpu')
    moment_alignment_model.load_state_dict(checkpoint)
    model_list[i] = moment_alignment_model.to(device)

prefix = '/Users/Johannes/Desktop/encoder_decoder_exp/' if local else prefix
suffix = '.pth'
encoder_decoder_model_paths = {
    'encoder_model_path': prefix + 'encoder_1_25_6_state_dict' + suffix,
    'decoder_model_path': prefix + 'decoder_1_25_6_state_dict' + suffix,
}

encoder = net.get_trained_encoder(encoder_decoder_model_paths)
decoder = net.get_trained_encoder(encoder_decoder_model_paths)

encoder.to(device)
decoder.to(device)

prefix = '../' if local else '/home/zenn/data/'
style_images_path = prefix + 'testset_style'
content_images_path = prefix + 'testset_content'

number_content_images = len(os.listdir(content_images_path))
number_style_images = len(os.listdir(style_images_path))

content_image_files = ['{}/{}'.format(content_images_path, os.listdir(content_images_path)[i])
                       for i in range(number_content_images)]
style_image_files = ['{}/{}'.format(style_images_path, os.listdir(style_images_path)[i])
                     for i in range(number_style_images)]

loader = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(256),
     transforms.ToTensor()])

content_images = [data_loader.image_loader(content_image_files[i], loader).to(device) for i in range(number_content_images)]
style_images = [data_loader.image_loader(style_image_files[i], loader).to(device) for i in range(number_style_images)]

prefix = './time_measurement_images/' if local else '/home/zenn/time_measurements/img_ma/'


def measure(model, last_moment, content_image, style_image, img_saving_number):
    # synchronize gpu time and measure fp
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        content_feature_map_batch = encoder(content_image)['r41'].view(512, 1, 32, 32)
        style_feature_map_batch = encoder(style_image)['r41'].view(512, 1, 32, 32)
        style_feature_map_batch_moments = utils.compute_moments_batches(style_feature_map_batch, last_moment=last_moment)
        content_feature_map_batch_moments = utils.compute_moments_batches(content_feature_map_batch, last_moment=last_moment)
        y_pred = decoder(model(content_feature_map_batch, content_feature_map_batch_moments,
                         style_feature_map_batch_moments, is_test=True).view(1, 512, 32, 32))
    torch.cuda.synchronize()
    elapsed_fp = time.time() - t0

    u.save_image(y_pred, prefix + 'img_{}.jpg'.format(img_saving_number))

    return elapsed_fp


def benchmark(model, last_moment):
    # dry runs
    for i in range(5):
        _ = measure(model, last_moment, content_images[i], style_images[i], -i)

    print('done with dry runs, now benchmarking')

    # start benchmarking
    t_forward = []
    img_saving_number = 0
    for i in range(number_content_images):
        for j in range(number_style_images):
            img_saving_number += 1
            t_fp = measure(model, last_moment, content_images[i], style_images[j], img_saving_number)
            t_forward.append(t_fp)

    # free memory
    del model

    return t_forward


def benchmark_models():
    for i in range(len(model_list)):
        print('benchmarking model {}'.format(model_path_list[i]))
        t = benchmark(model_list[i], i+2)
        print('forward pass: ', np.mean(np.asarray(t) * 1e3), '+/-', np.std(np.asarray(t) * 1e3))
        print('now getting the next model to evaluate')


if __name__ == '__main__':
    print('running main benchmarking loop')
    benchmark_models()
