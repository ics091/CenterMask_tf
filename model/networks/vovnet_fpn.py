import tensorflow as tf
import keras
import keras.layers as KL

VoVNet99_eSE_FPNStagesTo5 = {
    'config_stage_ch': [128, 160, 192, 224],
    'config_concat_ch': [256, 512, 768, 1024],
    'layer_per_block': 5,
    'block_per_stage': [1, 3, 9, 3],
    'eSE': True
}


def conv3x3(input_tensor, filter, kernel_size=3, strides=1):
    # Conv-BN-ReLu 3x3
    # filter == conv output channels
    x = input_tensor
    x = KL.Conv2D(filter,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  use_bias=False)(x)
    x = KL.BatchNormalization()(x)
    x = KL.ReLU()(x)
    return x


def conv1x1(input_tensor, filter, kernel_size=1, strides=1):
    # Conv-BN-ReLu 1x1
    # filter == conv output channels
    x = input_tensor
    x = KL.Conv2D(filter,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  use_bias=False)(x)
    x = KL.BatchNormalization()(x)
    x = KL.ReLU()(x)
    return x


def Hsigmoid(input_tensor):
    # in pytorch  nn.relu6
    x = input_tensor
    x = tf.nn.relu6(x + 3) / 6
    return x


def eSEModule(input_tensor, channel):
    x = input_tensor
    # in pytorch => nn.AdaptiveAvgPool2d(1)
    x = KL.GlobalAveragePooling2D()(x)
    x = tf.reshape(x, [x.shape[0], 1, 1, x.shape[1]])
    x = KL.Conv2D(channel, kernel_size=1, padding='valid')(x)
    x = Hsigmoid(x)
    return input_tensor * x


def _OSA_module(input_tensor,
                stage_ch,
                concat_ch,
                layer_per_block,
                SE=False,
                dcn=False):

    x = input_tensor
    output = []
    output.append(x)

    for i in range(layer_per_block):
        if dcn:
            # if dcn add
            pass
        else:
            x = conv3x3(x, stage_ch)
            output.append(x)

    # feature aggregation
    x = tf.concat(output, -1)
    x = conv1x1(x, concat_ch)

    # eSE
    x = eSEModule(x, concat_ch)
    return x


def _OSA_stage(input_tensor, stage_ch, concat_ch, block_per_stage,
               layer_per_block, stage_num, SE):
    x = input_tensor

    if not stage_num == 2:
        # in pytorch ceil_mode = True, maybe padding should set here
        x = tf.nn.max_pool2d(x, ksize=3, strides=2, padding='SAME')
    if block_per_stage != 1:
        SE = False
    x = _OSA_module(x, stage_ch, concat_ch, layer_per_block, SE=SE, dcn=False)

    for i in range(block_per_stage - 1):
        if i != block_per_stage - 2:  # last block
            SE = False
        x = _OSA_module(x,
                        stage_ch,
                        concat_ch,
                        layer_per_block,
                        SE=SE,
                        dcn=False)
    return x


def vovnet_graph(input_image, architecture, config):
    assert architecture in ["vovnet"]
    config_stage_ch = config['config_stage_ch']
    config_concat_ch = config['config_concat_ch']
    layer_per_block = config['layer_per_block']
    block_per_stage = config['block_per_stage']
    SE = config['eSE']

    x = input_image

    # Stem stage
    # stage 1
    x = conv3x3(x, 64, strides=2)
    x = conv3x3(x, 64, strides=1)
    x = conv3x3(x, 128, strides=2)

    # OSA stage
    aggr = []
    # stage (i + 2)
    for i in range(4):
        x = _OSA_stage(x, config_stage_ch[i], config_concat_ch[i],
                       block_per_stage[i], layer_per_block, i + 2, SE)
        print(x.shape)
        aggr.append(x)
    return aggr


def fpn(feature_maps, config, fpn_out_channel=256, top_blocks=None):

    # feature_maps => list[Tensor]

    out_channels = fpn_out_channel

    # feature pyramid layer
    _, C3, C4, C5 = feature_maps
    P5 = KL.Conv2D(out_channels, kernel_size=(1, 1))(C5)
    P4 = KL.Add(name="fpn_p4add")([
        KL.UpSampling2D(size=(2, 2))(P5),
        KL.Conv2D(out_channels, kernel_size=(1, 1))(C4)
    ])
    P3 = KL.Add(name="fpn_p3add")([
        KL.UpSampling2D(size=(2, 2))(P4),
        KL.Conv2D(out_channels, kernel_size=(1, 1))(C3)
    ])

    P3 = KL.Conv2D(out_channels, (3, 3), padding="SAME")(P3)
    P4 = KL.Conv2D(out_channels, (3, 3), padding="SAME")(P4)
    P5 = KL.Conv2D(out_channels, (3, 3), padding="SAME")(P5)

    P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2)(P5)
    P7 = KL.MaxPooling2D(pool_size=(1, 1), strides=2)(P6)

    fpn_maps = [P3, P4, P5, P6, P7]

    return fpn_maps


# FCOS
class Scale(tf.keras.layers.Layer):
    def __init__(self):
        super(Scale, self).__init__()

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape,
            dtype='float32',
            initializer='ones',
            trainable=True
        )

    def call(self, inputs):
        return inputs * self.scale


def FCOSHead(input_feature_maps, config, in_channels):
    x = input_feature_maps
    # num of classes
    num_classes = config.FCOS.NUM_CLASS - 1
    # conv num
    conv_num = config.FCOS.CONV_NUM

    logits = []
    bbox_reg = []
    centerness = []

    for l, feature in enumerate(x):
        cls_tower = feature
        for i in range(conv_num):
            cls_tower = KL.add()([
                KL.ZeroPadding2D(padding=(1, 1))(cls_tower),
                KL.Conv2D(in_channels, kernel_size=3, strides=1, padding="valid")(cls_tower)
            ])
            cls_tower = tf.contrib.layers.group_norm(cls_tower, groups=32, channels_axis=-1)
            cls_tower = KL.ReLU()(cls_tower)
        # logits
        logits.append(
            KL.add()([
                KL.ZeroPadding2D(padding=(1, 1))(cls_tower),
                KL.Conv2D(num_classes, kernel_size=3, strides=1, padding="valid")(cls_tower)
            ])
        )
        # centerness
        centerness.append(
            KL.add()([
                KL.ZeroPadding2D(padding=(1, 1))(cls_tower),
                KL.Conv2D(1, kernel_size=3, strides=1, padding="valid")(cls_tower)
            ])
        )
        scales = []
        for _ in range(5):
            scales.append(Scale)
        # bbox_reg
        bbox_tower = feature
        bbox_tower = KL.add()([
            KL.ZeroPadding2D(padding=(1, 1))(bbox_tower),
            KL.Conv2D(4, kernel_size=3, strides=1, padding="valid")(bbox_tower)
        ])
        bbox_tower = scales[l](bbox_tower)
        bbox_reg.append(tf.math.exp(bbox_tower))

    return logits, bbox_reg, centerness


class FCOSModule(tf.keras.layers.Layer):
    def __init__(self, config, in_channels):
        super(FCOSModule, self).__init__()
        self.cfg = config
        self.in_channels = in_channels

    def call(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)
        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression, centerness = FCOSHead(features, config=self.cfg, in_channels=self.in_channels)

    



if __name__ == "__main__":
    input = tf.random.normal((4, 1024, 1024, 4))
    # x = vovnet_graph(input, "vovnet", VoVNet99_eSE_FPNStagesTo5)
    # maps = fpn(x, VoVNet99_eSE_FPNStagesTo5)

    # x = tf.contrib.layers.group_norm(input, groups=2, channels_axis=-1)
