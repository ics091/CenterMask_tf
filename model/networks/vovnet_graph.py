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


def vovnet_graph(input_image, config):
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
        # print(x.shape)
        aggr.append(x)
    C2, C3, C4, C5 = aggr[0], aggr[1], aggr[2], aggr[3]
    return C2, C3, C4, C5


# if __name__ == "__main__":
#     input = tf.random.normal((4, 1024, 1024, 4))
#     C2, C3, C4, C5 = vovnet_graph(input, "vovnet", VoVNet99_eSE_FPNStagesTo5)
#     print(C2.shape)
#     print(C3.shape)
#     print(C4.shape)
#     print(C5.shape)
