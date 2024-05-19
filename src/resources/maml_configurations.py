########################################################
# conv2d: out, in, kernal(x, y), stride, padding
# bn (batch norm): out
# relu: inplace
# max_pool2d: kernal_size, stride, padding
# flatten: none
# linear: out, in
########################################################

from copy import deepcopy

ORIGINAL_MAML_CONFIG = [
    ("conv2d", [64, 1, 3, 3, 2, 0]),
    ("relu", [True]),
    ("bn", [64]),
    ("conv2d", [64, 64, 3, 3, 2, 0]),
    ("relu", [True]),
    ("bn", [64]),
    ("conv2d", [64, 64, 3, 3, 2, 0]),
    ("relu", [True]),
    ("bn", [64]),
    ("conv2d", [64, 64, 2, 2, 1, 0]),
    ("relu", [True]),
    ("bn", [64]),
    ("flatten", []),
    ("linear", [None, 8320]),
]

VGGNET_CONFIG = [
    ("conv2d", [16, 1, 3, 3, 1, 2]),
    ("bn", [16]),
    ("leakyrelu", [0.01, True]),
    ("max_pool2d", [2, 2, 0]),
    ###################
    ("conv2d", [32, 16, 3, 3, 1, 2]),
    ("bn", [32]),
    ("leakyrelu", [0.01, True]),
    ("max_pool2d", [2, 2, 0]),
    ###################
    ("conv2d", [64, 32, 3, 3, 1, 2]),
    ("bn", [64]),
    ("leakyrelu", [0.01, True]),
    ("max_pool2d", [2, 2, 0]),
    ###################
    ("conv2d", [128, 64, 3, 3, 1, 2]),
    ("bn", [128]),
    ("leakyrelu", [0.01, True]),
    ("max_pool2d", [2, 2, 0]),
    ###################
    ("flatten", []),
    ("linear", [None, 8064]),
]


def make_config(model: str, channels: int, num_classes: int):
    config = []
    if model == "original":
        config = deepcopy(ORIGINAL_MAML_CONFIG)
    else:
        config = deepcopy(VGGNET_CONFIG)

    config[0][1][1] = channels
    config[len(config) - 1][1][0] = num_classes

    return config
