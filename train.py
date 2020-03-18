import os
import sys
import torch
import torch.optim as optim
import torchvision.utils as u

# from tensorboardX import SummaryWriter

import utils
import data_loader
import net

# path to python_utils
sys.path.insert(0, '../utils')
sys.path.insert(0, '/home/zenn')

from python_utils.LossWriter import LossWriter


# device used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(configuration):
    """
    this is the main training loop
    :param configuration: the config file
    :return:
    """
    epochs = configuration['epochs']
    print('going to train for {} epochs'.format(epochs))

    step_printing_interval = configuration['step_printing_interval']
    print('writing to console every {} steps'.format(step_printing_interval))

    image_saving_interval = configuration['image_saving_interval']
    print('writing to console every {} steps'.format(step_printing_interval))

    epoch_saving_interval = configuration['epoch_saving_interval']
    print('saving the model every {} epochs'.format(epoch_saving_interval))

    validation_interval = configuration['validation_interval']
    print('validating the model every {} epochs'.format(validation_interval))

    image_saving_path = configuration['image_saving_path']
    print('saving images to {}'.format(image_saving_path))

    loader = configuration['loader']

    model_saving_path = configuration['model_saving_path']
    print('saving models to {}'.format(model_saving_path))

    # tensorboardX_path = configuration['tensorboardX_path']
    # writer = SummaryWriter(logdir='{}/runs'.format(tensorboardX_path))
    # print('saving tensorboardX logs to {}'.format(tensorboardX_path))

    loss_writer = LossWriter(os.path.join(configuration['folder_structure'].get_parent_folder(), './loss/loss'))
    loss_writer.write_header(columns=['epoch', 'all_training_iteration', 'loss', 'moment_loss', 'reconstruction_loss'])

    validation_loss_writer = LossWriter(os.path.join(configuration['folder_structure'].get_parent_folder(), './loss/loss'))
    validation_loss_writer.write_header(columns=['validation_iteration', 'loss', 'moment_loss', 'reconstruction_loss'])

    # batch_size is the number of images to sample
    batch_size = 1
    feature_map_batch_size = int(configuration['feature_map_batch_size'])
    print('training in batches of {} feature maps'.format(feature_map_batch_size))

    coco_data_path_train = configuration['coco_data_path_train']
    painter_by_numbers_data_path_train = configuration['painter_by_numbers_data_path_train']
    print('using {} and {} for training'.format(coco_data_path_train, painter_by_numbers_data_path_train))

    coco_data_path_val = configuration['coco_data_path_val']
    painter_by_numbers_data_path_val = configuration['painter_by_numbers_data_path_val']
    print('using {} and {} for validation'.format(coco_data_path_val, painter_by_numbers_data_path_val))

    train_dataloader = data_loader.get_concat_dataloader(
        coco_data_path_train, painter_by_numbers_data_path_train, batch_size, loader=loader)
    print('got train dataloader')

    val_dataloader = data_loader.get_concat_dataloader(
        coco_data_path_val, painter_by_numbers_data_path_val, batch_size, loader=loader)
    print('got val dataloader')

    lambda_1 = configuration['lambda_1']
    print('lambda 1: {}'.format(lambda_1))

    lambda_2 = configuration['lambda_2']
    print('lambda 2: {}'.format(lambda_2))

    loss_moment_mode = configuration['moment_mode']
    net_moment_mode = configuration['moment_mode']
    print('loss is sum of the first {} moments'.format(loss_moment_mode))
    print('net accepts {} in-channels'.format(net_moment_mode))

    unloader = configuration['unloader']
    print('got the unloader')

    do_validation = configuration['do_validation']
    print('doing validation: {}'.format(do_validation))

    moment_alignment_model = net.get_moment_alignment_model(configuration, moment_mode=loss_moment_mode)
    print('got model')
    print(moment_alignment_model)

    decoder = net.get_trained_decoder(configuration)
    print('got decoder')
    print(decoder)

    print('params that require grad')
    for name, param in moment_alignment_model.named_parameters():
        if param.requires_grad:
            print(name)

    criterion = net.get_loss(configuration, moment_mode=loss_moment_mode, lambda_1=lambda_1, lambda_2=lambda_2)

    print('got moment loss module')
    print(criterion)

    encoder = net.get_trained_encoder(configuration)
    print('got encoder')
    print(encoder)

    try:
        optimizer = optim.Adam(moment_alignment_model.parameters(), lr=configuration['lr'])
    except:
        optimizer = optim.Adam(moment_alignment_model.module.parameters(), lr=configuration['lr'])
    print('got optimizer')

    print('making iterable from train dataloader')
    train_data_loader = iter(train_dataloader)
    print('train data loader iterable')
    outer_training_iteration = -1
    all_training_iteration = 0

    number_of_validation = 0
    current_validation_loss = float('inf')

    for epoch in range(1, epochs):
        print('epoch: {}'.format(epoch))

        # this is the outer training loop (sampling images)
        print('training model ...')
        while True:
            try:
                data = train_data_loader.__next__()
                outer_training_iteration += 1
            except StopIteration:
                print('got to the end of the dataloader (StopIteration)')
                train_data_loader = iter(train_dataloader)
                break
            except:
                print('something went wrong with the dataloader, continuing')
                continue

            if do_validation:
                # validate the model every validation_interval iterations
                if outer_training_iteration % validation_interval == 0:
                    print('making iterable from val dataloader')
                    val_data_loader = iter(val_dataloader)
                    print('val data loader iterable')
                    validation_loss = validate(number_of_validation, criterion, encoder, moment_alignment_model,
                                               val_data_loader, feature_map_batch_size, validation_loss_writer)
                    number_of_validation += 1
                    if validation_loss < current_validation_loss:
                        utils.save_current_best_model(epoch, moment_alignment_model, configuration['model_saving_path'])
                        print('got a better model')
                        current_validation_loss = validation_loss
                        print('set the new validation loss to the current one')
                    else:
                        print('this model is actually worse than the best one')

            # get the content_image batch
            content_image = data.get('coco').get('image')
            content_image = content_image.to(device)

            # get the style_image batch
            style_image = data.get('painter_by_numbers').get('image')
            style_image = style_image.to(device)

            style_feature_maps = encoder(style_image)['r41'].to(device)
            content_feature_maps = encoder(content_image)['r41'].to(device)

            result_feature_maps = torch.zeros(1, 1, 32, 32)

            content_feature_map_batch_loader = data_loader.get_batch(content_feature_maps, feature_map_batch_size)
            style_feature_map_batch_loader = data_loader.get_batch(style_feature_maps, feature_map_batch_size)

            # this is the inner training loop (feature maps)
            while True:
                try:
                    content_feature_map_batch = next(content_feature_map_batch_loader).to(device)
                    style_feature_map_batch = next(style_feature_map_batch_loader).to(device)
                    all_training_iteration += 1
                except StopIteration:
                    break
                except:
                    continue

                do_print = all_training_iteration % step_printing_interval == 0

                optimizer.zero_grad()

                style_feature_map_batch_moments = utils.compute_moments_batches(style_feature_map_batch,
                                                                                last_moment=net_moment_mode)
                content_feature_map_batch_moments = utils.compute_moments_batches(content_feature_map_batch,
                                                                                  last_moment=net_moment_mode)

                out = moment_alignment_model(content_feature_map_batch,
                                             content_feature_map_batch_moments,
                                             style_feature_map_batch_moments)

                loss, moment_loss, reconstruction_loss = criterion(content_feature_map_batch,
                                                                   style_feature_map_batch,
                                                                   content_feature_map_batch_moments,
                                                                   style_feature_map_batch_moments,
                                                                   out,
                                                                   last_moment=loss_moment_mode)

                if do_print:
                    loss_writer.write_row([epoch, all_training_iteration, loss.item(), moment_loss.item(),
                                           reconstruction_loss.item()])

                # backprop
                loss.backward()
                optimizer.step()

                result_feature_maps = torch.cat([result_feature_maps, out.cpu().view(1, -1, 32, 32)], 1)

                # if do_print:
                #     print('loss: {:4f}'.format(loss.item()))
                #
                #     writer.add_scalar('data/training_loss', loss.item(), all_training_iteration)
                #     writer.add_scalar('data/training_moment_loss', moment_loss.item(), all_training_iteration)
                #     writer.add_scalar('data/training_reconstruction_loss', reconstruction_loss.item(),
                #                       all_training_iteration)

            result_feature_maps = result_feature_maps[:, 1:513, :, :]
            result_img = decoder(result_feature_maps.to(device))

            if outer_training_iteration % image_saving_interval == 0:
                u.save_image([data_loader.imnorm(content_image, unloader),
                              data_loader.imnorm(style_image, unloader),
                              data_loader.imnorm(result_img, None)],
                             '{}/image_{}_{}__{}_{}.jpeg'.format(
                                 image_saving_path, epoch, outer_training_iteration / epoch, lambda_1, lambda_2),
                             normalize=False)

            # save every epoch_saving_interval the current model
            if outer_training_iteration % image_saving_interval == 0:
                utils.save_current_model(lambda_1, lambda_2, moment_alignment_model.state_dict(),
                                         optimizer.state_dict(), configuration['model_saving_path'])


def validate(number_of_validation, criterion, encoder,
             moment_alignment_model, val_data_loader, feature_map_batch_size, writer):
    """
    the validaton loop
    :param number_of_validation: the number of the current validation (for data saving)
    :param criterion: the loss criterion
    :param encoder: the encoder network
    :param moment_alignment_model: the moment alignment model
    :param val_data_loader: the dataloader for the validation dataset
    :param feature_map_batch_size: the batch size
    :param writer: the (loss) writer
    :return:
    """
    print('validating model ...')
    iteration = 0
    total_validation_loss = 0
    while True:
        try:
            data = val_data_loader.__next__()
        except StopIteration:
            break
        except:
            print('something wrong happened with the data loader')
            continue

        # get the content_image batch
        content_image = data.get('coco').get('image')
        content_image = content_image.to(device)

        # get the style_image batch
        style_image = data.get('painter_by_numbers').get('image')
        style_image = style_image.to(device)

        with torch.no_grad():

            style_feature_maps = encoder(style_image)['r41'].to(device)
            content_feature_maps = encoder(content_image)['r41'].to(device)

            content_feature_map_batch_loader = data_loader.get_batch(content_feature_maps,
                                                                     feature_map_batch_size)
            style_feature_map_batch_loader = data_loader.get_batch(style_feature_maps, feature_map_batch_size)

            while True:
                try:
                    content_feature_map_batch = next(content_feature_map_batch_loader).to(device)
                    style_feature_map_batch = next(style_feature_map_batch_loader).to(device)
                    iteration += 1
                except StopIteration:
                    break
                except:
                    print('something wrong happened with the data loader')
                    continue

                style_feature_map_batch_moments = utils.compute_moments_batches(style_feature_map_batch)
                content_feature_map_batch_moments = utils.compute_moments_batches(content_feature_map_batch)

                out = moment_alignment_model(content_feature_map_batch,
                                             content_feature_map_batch_moments,
                                             style_feature_map_batch_moments)

                loss, moment_loss, reconstruction_loss = criterion(content_feature_map_batch,
                                                                       style_feature_map_batch,
                                                                       content_feature_map_batch_moments,
                                                                       style_feature_map_batch_moments,
                                                                       out)

                total_validation_loss += loss

                writer.write_row([number_of_validation * 100 * 512, loss.item(), moment_loss.item(),
                                       reconstruction_loss.item()])

                # if iteration % 2000 == 0:
                #     print('validation loss: {:4f}'.format(loss.item()))
                #     writer.add_scalar('data/validation_loss', loss.item(),
                #                       number_of_validation * 100 * 512 + iteration)
                #     writer.add_scalar('data/validation_moment_loss', moment_loss.item(),
                #                       number_of_validation * 100 * 512 + iteration)
                #     writer.add_scalar('data/validation_reconstruction_loss', reconstruction_loss.item(),
                #                       number_of_validation * 100 * 512 + iteration)

    writer.add_scalar('data/mean_validation_loss', total_validation_loss / iteration, number_of_validation)

    return total_validation_loss / iteration
