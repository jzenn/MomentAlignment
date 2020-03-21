import torch
import torch.nn as nn

import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


########################################################################
# models
########################################################################

class Encoder(nn.Module):
    """
    the encoder network to encode an image
    """
    def __init__(self):
        super(Encoder, self).__init__()
        # first block
        self.conv_1_1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.reflecPad_1_1 = nn.ReflectionPad2d((1, 1, 1, 1))

        self.conv_1_2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.relu_1_2 = nn.ReLU(inplace=True)

        self.reflecPad_1_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_1_3 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu_1_3 = nn.ReLU(inplace=True)

        self.maxPool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # second block
        self.reflecPad_2_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_1 = nn.Conv2d(64, 128, 3, 1, 0)
        self.relu_2_1 = nn.ReLU(inplace=True)

        self.reflecPad_2_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu_2_2 = nn.ReLU(inplace=True)

        self.maxPool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # third block
        self.reflecPad_3_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_1 = nn.Conv2d(128, 256, 3, 1, 0)
        self.relu_3_1 = nn.ReLU(inplace=True)

        self.reflecPad_3_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_3_2 = nn.ReLU(inplace=True)

        self.reflecPad_3_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_3_3 = nn.ReLU(inplace=True)

        self.reflecPad_3_4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_3_4 = nn.ReLU(inplace=True)

        self.maxPool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # fourth block
        self.reflecPad_4_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_4_1 = nn.Conv2d(256, 512, 3, 1, 0)
        self.relu_4_1 = nn.ReLU(inplace=True)

    def forward(self, input):
        output = {}

        # first block
        out = self.conv_1_1(input)
        out = self.reflecPad_1_1(out)
        out = self.conv_1_2(out)
        out = self.relu_1_2(out)

        output['r11'] = out

        out = self.reflecPad_1_3(out)
        out = self.conv_1_3(out)
        out = self.relu_1_3(out)

        out = self.maxPool_1(out)

        # second block
        out = self.reflecPad_2_1(out)
        out = self.conv_2_1(out)
        out = self.relu_2_1(out)

        output['r21'] = out

        out = self.reflecPad_2_2(out)
        out = self.conv_2_2(out)
        out = self.relu_2_2(out)

        out = self.maxPool_2(out)

        # third block
        out = self.reflecPad_3_1(out)
        out = self.conv_3_1(out)
        out = self.relu_3_1(out)

        output['r31'] = out

        out = self.reflecPad_3_2(out)
        out = self.conv_3_2(out)
        out = self.relu_3_2(out)

        out = self.reflecPad_3_3(out)
        out = self.conv_3_3(out)
        out = self.relu_3_3(out)

        out = self.reflecPad_3_4(out)
        out = self.conv_3_4(out)
        out = self.relu_3_4(out)

        out = self.maxPool_3(out)

        # fourth block
        out = self.reflecPad_4_1(out)
        out = self.conv_4_1(out)
        out = self.relu_4_1(out)

        output['r41'] = out

        return output


class Decoder(nn.Module):
    """
    the decoder network
    """
    def __init__(self):
        super(Decoder, self).__init__()
        # first block
        self.reflecPad_1_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_1_1 = nn.Conv2d(512, 256, 3, 1, 0)
        self.relu_1_1 = nn.ReLU(inplace=True)

        self.unpool_1 = nn.UpsamplingNearest2d(scale_factor=2)

        # second block
        self.reflecPad_2_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_1 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_2_1 = nn.ReLU(inplace=True)

        self.reflecPad_2_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_2_2 = nn.ReLU(inplace=True)

        self.reflecPad_2_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_2_3 = nn.ReLU(inplace=True)

        self.reflecPad_2_4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_4 = nn.Conv2d(256, 128, 3, 1, 0)
        self.relu_2_4 = nn.ReLU(inplace=True)

        self.unpool_2 = nn.UpsamplingNearest2d(scale_factor=2)

        # third block
        self.reflecPad_3_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_1 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu_3_1 = nn.ReLU(inplace=True)

        self.reflecPad_3_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_2 = nn.Conv2d(128, 64, 3, 1, 0)
        self.relu_3_2 = nn.ReLU(inplace=True)

        self.unpool_3 = nn.UpsamplingNearest2d(scale_factor=2)

        # fourth block
        self.reflecPad_4_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_4_1 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu_4_1 = nn.ReLU(inplace=True)

        self.reflecPad_4_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_4_2 = nn.Conv2d(64, 3, 3, 1, 0)

    def forward(self, input):
        # first block
        out = self.reflecPad_1_1(input)
        out = self.conv_1_1(out)
        out = self.relu_1_1(out)
        out = self.unpool_1(out)

        # second block
        out = self.reflecPad_2_1(out)
        out = self.conv_2_1(out)
        out = self.relu_2_1(out)
        out = self.reflecPad_2_2(out)
        out = self.conv_2_2(out)
        out = self.relu_2_2(out)
        out = self.reflecPad_2_3(out)
        out = self.conv_2_3(out)
        out = self.relu_2_3(out)
        out = self.reflecPad_2_4(out)
        out = self.conv_2_4(out)
        out = self.relu_2_4(out)
        out = self.unpool_2(out)

        # third block
        out = self.reflecPad_3_1(out)
        out = self.conv_3_1(out)
        out = self.relu_3_1(out)
        out = self.reflecPad_3_2(out)
        out = self.conv_3_2(out)
        out = self.relu_3_2(out)
        out = self.unpool_3(out)

        # fourth block
        out = self.reflecPad_4_1(out)
        out = self.conv_4_1(out)
        out = self.relu_4_1(out)
        out = self.reflecPad_4_2(out)
        out = self.conv_4_2(out)

        return out


def get_trained_encoder(configuration):
    """
    get a pre-trained encoder
    :param configuration: the config file
    :return:
    """
    encoder_model_path = configuration['encoder_model_path']
    print('loading encoder from {}'.format(encoder_model_path))

    checkpoint = torch.load(encoder_model_path, map_location='cpu')

    encoder = Encoder()
    encoder.load_state_dict(checkpoint)
    print('got encoder')

    for param in encoder.parameters():
        param.requires_grad = False

    return encoder.to(device)


def get_trained_decoder(configuration):
    """
    get a pre-trained decoder
    :param configuration: the config file
    :return:
    """
    decoder_model_path = configuration['decoder_model_path']
    print('loading decoder from {}'.format(decoder_model_path))
    print('loading decoder from {}'.format(decoder_model_path))

    checkpoint = torch.load(decoder_model_path, map_location='cpu')

    decoder = Decoder()
    decoder.load_state_dict(checkpoint)
    print('got decoder')

    for param in decoder.parameters():
        param.requires_grad = False

    return decoder.to(device)


class MomentAlignmentInceptionModule(nn.Module):
    """
    Moment Alignment Inception Module
    """
    def __init__(self, in_channels, out_channels):
        super(MomentAlignmentInceptionModule, self).__init__()

        self.relu = nn.ReLU()

        self.conv_1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.conv_2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv_3 = nn.Conv2d(in_channels, out_channels, 5, 1, 2)

        self.conv_4 = nn.Conv2d(out_channels * 3, out_channels, 1, 1, 0)

    def forward(self, input):
        out_1 = self.conv_1(input)
        out_1 = self.relu(out_1)

        out_2 = self.conv_2(input)
        out_2 = self.relu(out_2)

        out_3 = self.conv_3(input)
        out_3 = self.relu(out_3)

        return self.relu(self.conv_4(torch.cat([out_1, out_2, out_3], 1)))


class MomentAlignmentInceptionNetwork(nn.Module):
    """
    Moment Alignment Inception Network
    """
    def __init__(self, mode, in_channels):
        super(MomentAlignmentInceptionNetwork, self).__init__()
        self.mode = mode

        start_block = MomentAlignmentInceptionModule(in_channels=in_channels, out_channels=32)
        second_block = MomentAlignmentInceptionModule(in_channels=32, out_channels=16)
        inception_block = MomentAlignmentInceptionModule(in_channels=16, out_channels=16)
        end_block = MomentAlignmentInceptionModule(in_channels=16, out_channels=1)

        self.net = [start_block, second_block]

        for _ in range(self.mode - 2):
            self.net.append(inception_block)

        self.net.append(end_block)

        self.net = nn.ModuleList(self.net)

    def forward(self, feature_map, feature_map_moments, new_moments, do_print=False):
        # (n, c, h, w) content feature map size
        n, c, h, w = feature_map.size()

        # produce input to net
        for i in range(self.mode):
            target_moment_layer = new_moments[i].expand(n, c, h, w).to(device)
            input_moment_layer = feature_map_moments[i].expand(n, c, h, w).to(device)
            feature_map = torch.cat([feature_map, target_moment_layer, input_moment_layer], 1)

        # forward pass through inception blocks
        for i in range(self.mode):
            feature_map = self.net[i](feature_map)

        return feature_map.to(device)


class MomentAlignment(nn.Module):
    """
    Moment Alignment module
    """
    def __init__(self, mode, in_channels):
        super(MomentAlignment, self).__init__()
        self.mode = mode

        net = [
            nn.Conv2d(in_channels, 15, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(15, 10, 1, 1, 0),
            nn.ReLU()
        ]

        for _ in range(self.mode - 1):
            net.append(nn.Conv2d(10, 10, 1, 1, 0))
            net.append(nn.ReLU())

        net.append(nn.Conv2d(10, 1, 1, 1, 0))

        self.net = nn.Sequential(*net)

    def forward(self, feature_map, feature_map_moments, new_moments, do_print=False):

        # (n, c, h, w) content feature map size
        n, c, h, w = feature_map.size()

        # produce input to net
        for i in range(self.mode):
            target_moment_layer = new_moments[i].expand(n, c, h, w).to(device)
            input_moment_layer = feature_map_moments[i].expand(n, c, h, w).to(device)
            feature_map = torch.cat([feature_map, target_moment_layer, input_moment_layer], 1)

        # forward pass
        out = self.net(feature_map)

        return out.to(device)


class MomentAlignmentResidualNetwork(nn.Module):
    """
    Moment Alignment module
    """
    def __init__(self, mode, in_channels):
        super(MomentAlignmentResidualNetwork, self).__init__()
        self.mode = mode

        start_block = [
            nn.Conv2d(in_channels, 32, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(32, 16, 1, 1, 0),
            nn.ReLU()
        ]

        residual_block = [
            nn.Conv2d(16, 16, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(16, 16, 1, 1, 0),
            nn.ReLU()
        ]

        self.net = [nn.Sequential(*start_block)]

        for _ in range(self.mode - 1):
            self.net.append(nn.Sequential(*residual_block))

        self.net.append(nn.Conv2d(16, 1, 1, 1, 0))

        self.net = nn.ModuleList(self.net)

    def forward(self, feature_map, feature_map_moments, new_moments, do_print=False):
        # (n, c, h, w) content feature map size
        n, c, h, w = feature_map.size()

        # produce input to net
        for i in range(self.mode):
            target_moment_layer = new_moments[i].expand(n, c, h, w).to(device)
            input_moment_layer = feature_map_moments[i].expand(n, c, h, w).to(device)
            feature_map = torch.cat([feature_map, target_moment_layer, input_moment_layer], 1)

        # start block
        out = self.net[0](feature_map)

        # residual blocks
        for i in range(1, self.mode - 1):
            out = self.net[i](out) + out

        # end block
        out = self.net[-1](out)

        return out.to(device)


class AdaptiveInstanceNormalization(nn.Module):
    """
    Adaptive Instance Normaliztion (AdaIN) aligns the std and mean of two tensors channel-wise
    """
    def __init__(self):
        super(AdaptiveInstanceNormalization, self).__init__()

    def forward(self, input_content_features, input_style_features):
        assert (len(input_content_features.size()) == len(input_style_features.size())), \
            'the sizes of the content and style feature maps should be equal'

        feature_size = input_content_features.size()

        mean_content, std_content = utils.calc_mean_and_std(input_content_features)
        mean_style, std_style = utils.calc_mean_and_std(input_style_features)

        return std_style.expand(feature_size) \
               * (input_content_features - mean_content.expand(feature_size)) / std_content.expand(feature_size) \
               + mean_style.expand(feature_size)


def get_loss(configuration, moment_mode, lambda_1, lambda_2):
    """
    get the moment loss
    :param configuration: the config file
    :param moment_mode: the number of moments to be aligned
    :param lambda_1: weighting of the reconstruction loss
    :param lambda_2: weighting of the moment loss
    :return:
    """
    loss_module = AlignedMomentLoss(
        lambda_1=lambda_1, mode=moment_mode, lambda_2=lambda_2).to(device)
    return loss_module


def get_moment_alignment_model(configuration, moment_mode, use_list=False, list_index=False):
    """
    get the moment alignment module (pre-trained if specified)
    :param configuration: the config file
    :param moment_mode: the number of moments to be aligned
    :param use_list: whether to use multiple moment alignment modules
    :param list_index: the current index in the list
    :return:
    """
    use_pretrained_model = configuration['use_pretrained_model']

    if configuration['model'] == 'MomentAlignment':
        model = MomentAlignment(mode=moment_mode, in_channels=(2 * moment_mode + 1)).to(device)
    if configuration['model'] == 'MomentAlignmentResidualNetwork':
        model = MomentAlignmentResidualNetwork(mode=moment_mode, in_channels=(2 * moment_mode + 1)).to(device)
    if configuration['model'] == 'MomentAlignmentInceptionNetwork':
        model = MomentAlignmentInceptionNetwork(mode=moment_mode, in_channels=(2 * moment_mode + 1)).to(device)

    if use_list:
        model_path = configuration['model_path_list'][list_index]
    else:
        pretrained_model_path = configuration['pretrained_model_path']

    # use the max amount of GPUs possible
    if torch.cuda.device_count() > 1:
        print('Let\'s use', torch.cuda.device_count(), "GPUs!")
        model.to(device)
        model = nn.DataParallel(model)
    else:
        print('Let\'s use the {}'.format(device))

    if use_pretrained_model:
        if torch.cuda.device_count() <= 1:
            model = nn.DataParallel(model)
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print('loaded successfully model {}'.format(model_path))

    return model.to(device)


########################################################################
# loss
########################################################################

class AlignedMomentLoss(nn.Module):
    """
    the moment loss
    """
    def __init__(self, lambda_1, mode, lambda_2):
        super(AlignedMomentLoss, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.mode = mode
        self.l1_loss = nn.L1Loss()

    def forward(self, content_feature_map, style_feature_map, content_pre_moments, style_pre_moments, result_feature_map,
                do_print=False, last_moment=None, is_test=False):
        if last_moment is None or is_test is True:
            last_moment = self.mode

        # compute the moments to be aligned
        result_feature_map_moments = utils.compute_moments_batches(result_feature_map, last_moment=last_moment)

        # accumulate loss between target and input moments
        moment_loss = 0
        for i in range(len(result_feature_map_moments)):
            # set all entries |e_ij| < 1 to 1
            norm_factor = style_pre_moments[i].clone()
            norm_factor[abs(norm_factor) < 1] = 1
            norm_factor = 1 / torch.mean(norm_factor)

            current_moment_loss = norm_factor * self.l1_loss(result_feature_map_moments[i], style_pre_moments[i])

            moment_loss += current_moment_loss

        # reconstruction loss between result feature map and content feature map
        reconstruction_loss = self.l1_loss(result_feature_map, content_feature_map)

        if do_print:
            print('reconstruction loss: {:4f}, moment loss: {:4f}'.format(self.lambda_1 * reconstruction_loss, self.lambda_2 * moment_loss))

        loss = self.lambda_1 * reconstruction_loss + self.lambda_2 * moment_loss

        return loss.to(device), moment_loss, reconstruction_loss
