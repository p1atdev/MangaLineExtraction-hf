from collections import OrderedDict
from typing import Any, List, Mapping, Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.onnx import OnnxConfigWithPast, PatchingSpec
from transformers.utils import logging


logger = logging.get_logger(__name__)


class MLEConfig(PretrainedConfig):
    model_type = "mle"

    def __init__(
        self,
        in_channels=1,
        num_encoder_layers=[2, 3, 5, 7, 12],
        num_decoder_layers=[7, 5, 3, 2, 2],
        last_hidden_channels=16,
        block_stride_size=4,
        block_kernel_size=3,
        block_patch_size=24,
        upsample_ratio=2,
        batch_norm_eps=1e-3,
        hidden_act="leaky_relu",
        negative_slope=0.2,
        **kwargs,
    ):
        self.in_channels = in_channels
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.last_hidden_channels = last_hidden_channels

        self.block_stride_size = block_stride_size
        # if isinstance(block_kernel_size, int):
        #     self.block_kernel_size = (block_kernel_size, block_kernel_size)
        self.block_kernel_size = block_kernel_size
        self.block_patch_size = block_patch_size

        self.upsample_ratio = upsample_ratio
        self.batch_norm_eps = batch_norm_eps
        self.hidden_act = hidden_act
        self.negative_slope = negative_slope

        super().__init__(**kwargs)
