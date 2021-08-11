# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import warnings

import torch
from torch import nn

from src.segmentation.models.model import ESANet
from src.segmentation.models.model_one_modality import ESANetOneModality
from src.segmentation.models.resnet import ResNet


def build_model(n_classes):
    pretrained_on_imagenet = False
    last_ckpt = ''
    decoder_channels_mode = 'decreasing'
    channels_decoder = 128
    nr_decoder_blocksa = [3]
    nr_decoder_blocks = [3]
    modality = 'rgbd'
    encoder_depth = None
    encoder = 'resnet34'
    he_init = False
    pretrained_scenenet = ''
    context_module = 'ppm'
    finetune = None
    height=640
    width=480
    activation='relu'
    encoder_block='NonBottleneck1D'
    encoder_decoder_fusion='add'
    fuse_depth_in_rgb_encoder='SE-add'
    upsampling='learned-3x3-zeropad'
    dataset ='sunrgbd'
    dataset_dir= './datasets/sunrgbd'
    pretrained_dir= '/home/chris/GAN_SLAM/src/segmentation/trained_models/nyuv2/r34_NBt1D_scenenet.pth'
    batch_size= 4

    if not pretrained_on_imagenet or last_ckpt or \
            pretrained_scenenet != '':
        pretrained_on_imagenet = False
    else:
        pretrained_on_imagenet = True

    # set the number of channels in the encoder and for the
    # fused encoder features
    if 'decreasing' in decoder_channels_mode:
        if decoder_channels_mode == 'decreasing':
            channels_decoder = [512, 256, 128]

        warnings.warn('Argument --channels_decoder is ignored when '
                      '--decoder_chanels_mode decreasing is set.')
    else:
        channels_decoder = [channels_decoder] * 3

    if isinstance(nr_decoder_blocksa, int):
        nr_decoder_blocks = [nr_decoder_blocksa] * 3
    elif len(nr_decoder_blocksa) == 1:
        nr_decoder_blocks = nr_decoder_blocksa * 3
    else:
        nr_decoder_blocks = nr_decoder_blocksa
        assert len(nr_decoder_blocks) == 3

    if modality == 'rgbd':
        # use the same encoder for depth encoder and rgb encoder if no
        # specific depth encoder is provided
        if encoder_depth in [None, 'None']:
            encoder_depth = encoder

        model = ESANet(
            height=height,
            width=width,
            num_classes=n_classes,
            pretrained_on_imagenet=pretrained_on_imagenet,
            pretrained_dir=pretrained_dir,
            encoder_rgb=encoder,
            encoder_depth=encoder_depth,
            encoder_block=encoder_block,
            activation=activation,
            encoder_decoder_fusion=encoder_decoder_fusion,
            context_module=context_module,
            nr_decoder_blocks=nr_decoder_blocks,
            channels_decoder=channels_decoder,
            fuse_depth_in_rgb_encoder=fuse_depth_in_rgb_encoder,
            upsampling=upsampling
        )

    else:  # just one modality
        if modality == 'rgb':
            input_channels = 3
        else:  # depth only
            input_channels = 1

        model = ESANetOneModality(
            height=height,
            width=width,
            pretrained_on_imagenet=pretrained_on_imagenet,
            encoder=encoder,
            encoder_block=encoder_block,
            activation=activation,
            input_channels=input_channels,
            encoder_decoder_fusion=encoder_decoder_fusion,
            context_module=context_module,
            num_classes=n_classes,
            pretrained_dir=pretrained_dir,
            nr_decoder_blocks=nr_decoder_blocks,
            channels_decoder=channels_decoder,
            weighting_in_encoder=fuse_depth_in_rgb_encoder,
            upsampling=upsampling
        )

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print('Device:', device)
    model.to(device)
    print(model)

    if he_init:
        module_list = []

        # first filter out the already pretrained encoder(s)
        for c in model.children():
            if pretrained_on_imagenet and isinstance(c, ResNet):
                # already initialized
                continue
            for m in c.modules():
                module_list.append(m)

        # iterate over all the other modules
        # output layers, layers followed by sigmoid (in SE block) and
        # depthwise convolutions (currently only used in learned upsampling)
        # are not initialized with He method
        for i, m in enumerate(module_list):
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                if m.out_channels == n_classes or \
                        isinstance(module_list[i+1], nn.Sigmoid) or \
                        m.groups == m.in_channels:
                    continue
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        print('Applied He init.')

    if pretrained_scenenet != '':
        checkpoint = torch.load(pretrained_scenenet)

        weights_scenenet = checkpoint['state_dict']

        # (side) outputs and learned upsampling
        keys_to_ignore = [
            k for k in weights_scenenet
            if 'out' in k or 'decoder.upsample1' in k or 'decoder.upsample2' in k
        ]
        if context_module not in ['ppm', 'appm']:
            keys_to_ignore.extend([k for k in weights_scenenet
                                   if 'context_module.features' in k])

        for key in keys_to_ignore:
            weights_scenenet.pop(key)

        weights_model = model.state_dict()

        # just for verification that weight loading/ updating works
        # import copy
        # weights_before = copy.deepcopy(weights_model)

        weights_model.update(weights_scenenet)
        model.load_state_dict(weights_model)

        print(f'Loaded pretrained SceneNet weights: {args.pretrained_scenenet}')

    if finetune is not None:
        checkpoint = torch.load(finetune)
        model.load_state_dict(checkpoint['state_dict'])
        print(f'Loaded weights for finetuning: {args.finetune}')

        print('Freeze the encoder(s).')
        for name, param in model.named_parameters():
            if 'encoder_rgb' in name or 'encoder_depth' in name or 'se_layer' in name:
                param.requires_grad = False

    return model, device
