import tensorflow as tf
import keras
import keras.layers as KL
from . import vovnet_graph

from models import retinanet
from models import Backbone
from utils.image import preprocess_image


VoVNet99_eSE_FPNStagesTo5 = {
    'config_stage_ch': [128, 160, 192, 224],
    'config_concat_ch': [256, 512, 768, 1024],
    'layer_per_block': 5,
    'block_per_stage': [1, 3, 9, 3],
    'eSE': True
}


class VoVNetBackbone(Backbone):
    def __init__(self, backbone):
        super(VoVNetBackbone, self).__init__(backbone)

    def retinanet(self, *args, **kwargs):
        """
        Returns a retinanet model using the correct backbone.
        """
        return vovnet_retinanet(*args, backbone=self.backbone, **kwargs)

    def validate(self):
        """
        Checks whether the backbone string is correct.
        """
        allowed_backbones = ['vovnet']

        if self.backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(self.backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """
        Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='caffe')


def vovnet_retinanet(num_classes, backbone='vovnet', inputs=None, modifier=None, **kwargs):
    # choose default input
    if inputs is None:
        if keras.backend.image_data_format() == 'channels_first':
            inputs = keras.layers.Input(shape=(3, None, None))
        else:
            inputs = keras.layers.Input(shape=(None, None, 3))

    # create the resnet backbone
    if backbone == 'vovnet':
        vovnet = vovnet_graph(vovnet_graph(inputs, "vovnet", VoVNet99_eSE_FPNStagesTo5))
        # vovnet 返回值是 [C2, C3, C4, C5]
        return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=vovnet[1:], **kwargs)
