import torch

import torchvision.transforms as transforms

import sys
import yaml
import pprint

import train
import test
import test_analytical_solution
import test_multiple

# the device used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


########################################################################
# configuration loading
########################################################################

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.SafeLoader)


configuration = get_config(sys.argv[1])
action = configuration['action']
print('the configuration used is:')
pprint.pprint(configuration, indent=4)


########################################################################
# image loaders and unloaders
########################################################################

# image size
imsize = configuration['imsize']

# loaders
loaders = {
    'std':      transforms.Compose(
                    [transforms.Resize(imsize),
                     transforms.RandomResizedCrop(256),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    'no_norm':  transforms.Compose(
                    [transforms.Resize(imsize),
                     transforms.RandomResizedCrop(256),
                     transforms.ToTensor()])
}

# unloaders
unloaders = {
    'std':      transforms.Compose(
                    [transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
                     transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
                     transforms.ToPILImage()]),
    'saving':   transforms.Compose(
                    [transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
                     transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])]),
    'no_norm':  transforms.Compose(
                    [transforms.ToPILImage()]),
    'none':     None
}

configuration['loader'] = loaders[configuration['loader']]
configuration['unloader'] = unloaders[configuration['unloader']]

########################################################################
# main method
########################################################################

if __name__ == '__main__':
    if action == 'train':
        print('starting main training loop with specified configuration')
        train.train(configuration)
    elif action == 'test':
        print('starting test loop for content and style images')
        test.test(configuration)
    elif action == 'test_analytical_solution':
        print('starting test loop for content and style images')
        test_analytical_solution.test(configuration)
    elif action == 'test_multiple':
        print('starting test compare loop for moments 12 ... 5')
        test_multiple.test(configuration)


# # parameters
# action = sys.argv[1]
# data_path = sys.argv[2]
# working_directory = sys.argv[3]
# lambda_1 = int(sys.argv[4])
# lambda_2 = int(sys.argv[5])
# moment_mode = int(sys.argv[6])
# loss_mode = int(sys.argv[7])
# loss_moment_mode = 3
#
# try:
#     do_incremental_training = sys.argv[8] == 'True'
# except:
#     do_incremental_training = False
#
# try:
#     feature_map_batch_size = int(sys.argv[9])
# except:
#     feature_map_batch_size = 8
#
# try:
#     fix_weights_during_stable = sys.argv[10] == 'True'
# except:
#     fix_weights_during_stable = False
#
# try:
#     use_big_dataset = sys.argv[11] == 'True'
# except:
#     use_big_dataset = False
#
# try:
#     use_very_shallow_model = sys.argv[12] == 'True'
# except:
#     use_very_shallow_model = False
#
# try:
#     do_validation = sys.argv[13] == 'True'
# except:
#     do_validation = False
#
# try:
#     load_moment_alignment_model = sys.argv[14] == 'True'
# except:
#     load_moment_alignment_model = True
#
# try:
#     normalize_loss = sys.argv[15] == 'True'
# except:
#     normalize_loss = True
#
# model_path = '{}/../bachelor_thesis/pytorch_models/mom_al_model_{}'.format(data_path, moment_mode)
#
# increment_every_epoch = 5
#
# # mscoco dataset
# coco_data_path = '{}/cocodataset/train2017'.format(data_path)
# coco_data_path_train = '{}/cocodataset/train_7000'.format(data_path)
# coco_data_path_test = '{}/cocodataset/test_2000'.format(data_path)
# coco_data_path_val = '{}/cocodataset/val_1000'.format(data_path)
#
# if use_big_dataset:
#     coco_data_path_train = '{}/cocodataset/train_80000'.format(data_path)
#     coco_data_path_val = '{}/cocodataset/val_10000'.format(data_path)
#
# # painter-by-numbers dataset
# painter_by_numbers_data_path = '{}/painter-by-numbers/train'.format(data_path)
# painter_by_numbers_data_path_train = '{}/painter-by-numbers/train_7000'.format(data_path)
# painter_by_numbers_data_path_test = '{}/painter-by-numbers/test_2000'.format(data_path)
# painter_by_numbers_data_path_val = '{}/painter-by-numbers/val_1000'.format(data_path)
#
# if use_big_dataset:
#     painter_by_numbers_data_path_train = '{}/painter-by-numbers/train_80000'.format(data_path)
#     painter_by_numbers_data_path_val = '{}/painter-by-numbers/val_10000'.format(data_path)
#
# # model saving
# model_saving_path = '{}/moment_alignment_models_{}_{}_{}_{}'.format(working_directory, lambda_1,
#                                                                     lambda_2, moment_mode, loss_mode)
# image_saving_path = '{}/moment_alignment_images_{}_{}_{}_{}'.format(working_directory, lambda_1,
#                                                                     lambda_2, moment_mode, loss_mode)
#
# # tensorboardX log saving
# tensorboardX_path = '{}/moment_alignment_run_{}_{}_{}_{}'.format(working_directory, lambda_1,
#                                                                  lambda_2, moment_mode, loss_mode)
#
# # image size
# imsize = 512 if device == 'cuda' else 256  # size of images
#
# # image loaders
# loader = transforms.Compose(
#     [transforms.Resize(imsize),
#      transforms.RandomResizedCrop(256),
#      transforms.ToTensor(),
#      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#
# test_loader = transforms.Compose(
#     [transforms.Resize(256),
#      transforms.CenterCrop(256),
#      transforms.ToTensor(),
#      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#
# saving_loader_without_norm = transforms.Compose(
#     [transforms.Resize(imsize),
#      transforms.RandomResizedCrop(256),
#      transforms.ToTensor()])
#
# lst_loader = transforms.Compose(
#     [transforms.Resize(256),
#      transforms.CenterCrop(256),
#      transforms.ToTensor()])
#
# # image unloaders
# unloader = transforms.Compose(
#     [transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
#      transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
#      transforms.ToPILImage()])
#
# saving_unloader = transforms.Compose(
#     [transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
#      transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])])
#
# saving_unloader_without_norm = transforms.Compose([transforms.ToPILImage()])
#
# lst_unloader = None
#
# # live_encoder_model_path = '{}/../bachelor_thesis/pytorch_models/encoder_1_10__30_05'.format(data_path)
# # live_decoder_model_path = '{}/../bachelor_thesis/pytorch_models/decoder_1_10__30_05'.format(data_path)
#
# live_encoder_model_path = '{}/../encoder_decoder_balancing_lua/encoder_decoder_models_1_20_False/encoder.pth'.format(data_path)
# live_decoder_model_path = '{}/../encoder_decoder_balancing_lua/encoder_decoder_models_1_20_False/decoder.pth'.format(data_path)
#
# # configurations
# # for testing purposes
# train_configuration_test = {
#     'epochs': 100,
#     'epoch_saving_interval': 1,
#     'step_printing_interval': 1,
#     'image_saving_interval': 100,
#     'validation_interval': 200,
#     'coco_data_path': coco_data_path,
#     'coco_data_path_train': coco_data_path,
#     'coco_data_path_test': coco_data_path,
#     'coco_data_path_val': coco_data_path,
#     'painter_by_numbers_data_path': painter_by_numbers_data_path,
#     'painter_by_numbers_data_path_train': painter_by_numbers_data_path,
#     'painter_by_numbers_data_path_test': painter_by_numbers_data_path,
#     'painter_by_numbers_data_path_val': painter_by_numbers_data_path,
#     'model_saving_path': model_saving_path,
#     'image_saving_path': image_saving_path,
#     'tensorboardX_path': tensorboardX_path,
#     'batch_size': 1,
#     'feature_map_batch_size': feature_map_batch_size,
#     'loader': lst_loader,
#     'unloader': lst_unloader,
#     'lr': 0.0001,
#     'load_model': load_moment_alignment_model,
#     'model_path': '../pytorch_models/mom_al_model_{}'.format(moment_mode),
#     'url': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
#     'model_dir': '{}/torchvision_models/'.format(data_path),
#     'lambda_1': lambda_1,
#     'lambda_2': lambda_2,
#     'moment_mode': moment_mode,
#     'loss_moment_mode': loss_moment_mode,
#     'loss_mode': loss_mode,
#     'fix_weights_during_stable': fix_weights_during_stable,
#     'decoder_model_path': '/Users/Johannes/Desktop/clean/1_20/decoder.pth',
#     'encoder_model_path': '/Users/Johannes/Desktop/clean/1_20/encoder.pth',
#     'do_incremental_training': do_incremental_training,
#     'increment_every_epoch': increment_every_epoch,
#     'use_very_shallow_model': use_very_shallow_model,
#     'do_validation': do_validation,
#     'normalize_loss': normalize_loss
# }
#
# # for the actual training
# train_configuration = {
#     'epochs': 100,
#     'epoch_saving_interval': 1,
#     'step_printing_interval': 5000,
#     'image_saving_interval': 200,
#     'validation_interval': 50000,
#     'coco_data_path': coco_data_path,
#     'coco_data_path_train': coco_data_path_train,
#     'coco_data_path_test': coco_data_path_test,
#     'coco_data_path_val': coco_data_path_val,
#     'painter_by_numbers_data_path': painter_by_numbers_data_path,
#     'painter_by_numbers_data_path_train': painter_by_numbers_data_path_train,
#     'painter_by_numbers_data_path_test': painter_by_numbers_data_path_test,
#     'painter_by_numbers_data_path_val': painter_by_numbers_data_path_val,
#     'model_saving_path': model_saving_path,
#     'image_saving_path': image_saving_path,
#     'tensorboardX_path': tensorboardX_path,
#     'batch_size': 1,
#     'feature_map_batch_size': feature_map_batch_size,
#     'loader': lst_loader,
#     'unloader': lst_unloader,
#     'lr': 0.0001,
#     'load_model': load_moment_alignment_model,
#     'model_path': model_path,
#     'url': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
#     'model_dir': '{}/torchvision_models/'.format(data_path),
#     'lambda_1': lambda_1,
#     'lambda_2': lambda_2,
#     'moment_mode': moment_mode,
#     'loss_moment_mode': loss_moment_mode,
#     'loss_mode': loss_mode,
#     'fix_weights_during_stable': fix_weights_during_stable,
#     'encoder_model_path': live_encoder_model_path,
#     'decoder_model_path': live_decoder_model_path,
#     'do_incremental_training': do_incremental_training,
#     'increment_every_epoch': increment_every_epoch,
#     'use_very_shallow_model': use_very_shallow_model,
#     'do_validation': do_validation,
#     'normalize_loss': normalize_loss
# }
#
# style_images_path = '../testset_style'
# content_images_path = '../testset_content'
# # style_images_path = '/Users/Johannes/Desktop/tree_test_style'
# # content_images_path = '/Users/Johannes/Desktop/tree_test'
# # model_path = '../pytorch_models/mom_al_model_4'
# #model_path = '/Users/Johannes/Desktop/moment_alignment_model_3_mom'     # ###!
# model_path = '/Users/Johannes/Desktop/mom_al_shallow/moment_alignment_model_0_1_2_0'     # ###!
# #decoder_model_path = '../pytorch_models/decoder_1_50.pth'
# #encoder_model_path = '../pytorch_models/encoder_1_50.pth'
#
# # path is specified in the function in linear_style_transfer_py (this does do nothing)
# # encoder_model_path = '/Users/Johannes/Desktop/1_1/encoder.pth'         # ###!
# # decoder_model_path = '/Users/Johannes/Desktop/1_1/decoder.pth'         # ###!
#
# encoder_model_path = '/Users/Johannes/Desktop/encoder_decoder_exp/encoder_1_25_state_dict.pth'         # ###!
# decoder_model_path = '/Users/Johannes/Desktop/encoder_decoder_exp/decoder_1_25_ state_dict.pth'         # ###!
#
#
# image_test_saving_path = './moment_alignment_images_compare_analytical'
# use_very_shallow_model = False   # ###!
#
# prefix = '/Users/Johannes/Desktop/mom_al_deep/'
# suffix = ''
# model_path_list = [
#     prefix + 'moment_alignment_model_0_1_2_0' + suffix,
#     prefix + 'moment_alignment_model_0_1_3_0' + suffix,
#     prefix + 'moment_alignment_model_0_1_4_0' + suffix,
#     prefix + 'moment_alignment_model_0_1_5_0' + suffix,
# ]
#
# mode_list = [
#     2,
#     3,
#     4,
#     5
# ]
#
# use_very_shallow_model_list = [False, False, False, False]
#
# # use_very_shallow_model_list = [True, True, True, True, True, True]
#
# image_test_saving_path = './result_images_ada_in_ba'
# style_images_path = '/Users/Johannes/Desktop/ba_t_s'
# content_images_path = '/Users/Johannes/Desktop/ba_t_c'
# encoder_model_path = '/Users/Johannes/Desktop/encoder_decoder_exp/encoder_1_25_6_state_dict.pth'
# decoder_model_path = '/Users/Johannes/Desktop/encoder_decoder_exp/decoder_1_25_6_state_dict.pth'
# model_path = '/Users/Johannes/Desktop/mom_al_deep/moment_alignment_model_0_1_2_0'
#
# # the test configuration
# test_configuration = {
#     'style_images_path': style_images_path,
#     'content_images_path': content_images_path,
#     'image_saving_path': image_test_saving_path,
#     'decoder_model_path': decoder_model_path,
#     'encoder_model_path': encoder_model_path,
#     'model_path': model_path,
#     'model_path_list': model_path_list,
#     'lst_loader': lst_loader,
#     'lst_unloader': lst_unloader,
#     'loader': lst_loader,
#     'unloader': lst_unloader,
#     'saving_unloader': saving_unloader_without_norm,
#     'saving_loader_without_norm': saving_loader_without_norm,
#     'lr': 0.001,
#     'mode': 2,                                                          # ###!
#     'mode_list': mode_list,
#     'model_dir': '/Users/Johannes/',
#     'url': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
#     'do_incremental_training': False,
#     'load_model': False,
#     'use_very_shallow_model': use_very_shallow_model,
#     'use_very_shallow_model_list': use_very_shallow_model_list,
#     'normalize_loss': True
# }
#
# if __name__ == '__main__':
#     if action == 'train':
#         print('starting main training loop with specified configuration')
#         train.train(train_configuration)
#     if action == 'train_test':
#         print('starting main training loop with test configuration')
#         train.train(train_configuration_test)
#     if action == 'train_incremental_loss':
#         print('starting main training loop (incremental loss) with specified configuration')
#         train.train_loss_incremental(train_configuration)
#     if action == 'train_test_incremental_loss':
#         print('starting main training loop (incremental loss) with test configuration')
#         train.train_loss_incremental(train_configuration_test)
#     elif action == 'test':
#         print('starting test loop for content and style images')
#         test.test(test_configuration)
#     elif action == 'test_analytical_solution':
#         print('starting test loop for content and style images')
#         test_analytical_solution.test(test_configuration)
#     elif action == 'test_compare':
#         print('starting test compare loop for moments 12 ... 5')
#         test_compare.test(test_configuration)
#     elif action == 'testloop':
#         for network in ['deep', 'shallow']:
#         # for network in ['shallow']:
#         #     for i in range(3, 7):
#             for i in [5]:
#                 ma_path = '/Users/Johannes/Desktop/ma/mom_al_{}_{}'.format(i, network)
#                 image_saving_path = './result_images_{}_{}'.format(network, i)
#                 test_configuration['model_path'] = ma_path
#                 test_configuration['image_saving_path'] = image_saving_path
#                 test_configuration['mode'] = i
#                 if network == 'shallow':
#                     test_configuration['use_very_shallow_model'] = True
#                 test.test(test_configuration)
#
#
