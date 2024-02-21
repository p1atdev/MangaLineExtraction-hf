"""PyTorch MLE (Mnaga Line Extraction) model"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput, BaseModelOutput
from transformers.activations import ACT2FN

from .configuration_mle import MLEConfig


@dataclass
class MLEModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = None


@dataclass
class MLEForAnimeLineExtractionOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = None
    pixel_values: torch.Tensor | None = None


class MLEBatchNorm(nn.Module):
    def __init__(
        self,
        config: MLEConfig,
        in_features: int,
    ):
        super().__init__()

        self.norm = nn.BatchNorm2d(in_features, eps=config.batch_norm_eps)
        # the original model uses leaky_relu
        if config.hidden_act == "leaky_relu":
            self.act_fn = nn.LeakyReLU(negative_slope=config.negative_slope)
        else:
            self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        hidden_states = self.act_fn(hidden_states)

        return hidden_states


class MLEResBlock(nn.Module):
    def __init__(
        self,
        config: MLEConfig,
        in_channels: int,
        out_channels: int,
        stride_size: int,
    ):
        super().__init__()

        self.norm1 = MLEBatchNorm(config, in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            config.block_kernel_size,
            stride=stride_size,
            padding=config.block_kernel_size // 2,
        )

        self.norm2 = MLEBatchNorm(config, out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            config.block_kernel_size,
            stride=1,
            padding=config.block_kernel_size // 2,
        )

        if in_channels != out_channels or stride_size != 1:
            self.resize = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride_size,
            )
        else:
            self.resize = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.norm1(hidden_states)
        output = self.conv1(output)
        output = self.norm2(output)
        output = self.conv2(output)

        if self.resize is not None:
            resized_input = self.resize(hidden_states)
            output += resized_input
        else:
            output += hidden_states

        return output


class MLEEncoderLayer(nn.Module):
    def __init__(
        self,
        config: MLEConfig,
        in_features: int,
        out_features: int,
        num_layers: int,
        stride_sizes: list[int],
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                MLEResBlock(
                    config,
                    in_channels=in_features if i == 0 else out_features,
                    out_channels=out_features,
                    stride_size=stride_sizes[i],
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class MLEEncoder(nn.Module):
    def __init__(
        self,
        config: MLEConfig,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                MLEEncoderLayer(
                    config,
                    in_features=(
                        config.in_channels
                        if i == 0
                        else config.in_channels
                        * config.block_patch_size
                        * (config.upsample_ratio ** (i - 1))
                    ),
                    out_features=config.in_channels
                    * config.block_patch_size
                    * (config.upsample_ratio**i),
                    num_layers=num_layers,
                    stride_sizes=(
                        [
                            1 if i_layer < num_layers - 1 else 2
                            for i_layer in range(num_layers)
                        ]
                        if i > 0
                        else [1 for _ in range(num_layers)]
                    ),
                )
                for i, num_layers in enumerate(config.num_encoder_layers)
            ]
        )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        all_hidden_states: tuple[torch.Tensor, ...] = ()
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            all_hidden_states += (hidden_states,)
        return hidden_states, all_hidden_states


class MLEUpsampleBlock(nn.Module):
    def __init__(self, config: MLEConfig, in_features: int, out_features: int):
        super().__init__()

        self.norm = MLEBatchNorm(config, in_features=in_features)
        self.conv = nn.Conv2d(
            in_features,
            out_features,
            config.block_kernel_size,
            stride=1,
            padding=config.block_kernel_size // 2,
        )
        self.upsample = nn.Upsample(scale_factor=config.upsample_ratio)

    def forward(self, hidden_states: torch.Tensor):
        output = self.norm(hidden_states)
        output = self.conv(output)
        output = self.upsample(output)

        return output


class MLEUpsampleResBlock(nn.Module):
    def __init__(self, config: MLEConfig, in_features: int, out_features: int):
        super().__init__()

        self.upsample = MLEUpsampleBlock(
            config, in_features=in_features, out_features=out_features
        )

        self.norm = MLEBatchNorm(config, in_features=out_features)
        self.conv = nn.Conv2d(
            out_features,
            out_features,
            config.block_kernel_size,
            stride=1,
            padding=config.block_kernel_size // 2,
        )

        if in_features != out_features:
            self.resize = nn.Sequential(
                nn.Conv2d(
                    in_features,
                    out_features,
                    kernel_size=1,
                    stride=1,
                ),
                nn.Upsample(scale_factor=config.upsample_ratio),
            )
        else:
            self.resize = None

    def forward(self, hidden_states: torch.Tensor):
        output = self.upsample(hidden_states)
        output = self.norm(output)
        output = self.conv(output)

        if self.resize is not None:
            output += self.resize(hidden_states)

        return output


class MLEDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MLEConfig,
        in_features: int,
        out_features: int,
        num_layers: int,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                (
                    MLEResBlock(
                        config,
                        in_channels=out_features,
                        out_channels=out_features,
                        stride_size=1,
                    )
                    if i > 0
                    else MLEUpsampleResBlock(
                        config,
                        in_features=in_features,
                        out_features=out_features,
                    )
                )
                for i in range(num_layers)
            ]
        )

    def forward(
        self, hidden_states: torch.Tensor, shortcut_states: torch.Tensor
    ) -> torch.Tensor:
        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states += shortcut_states

        return hidden_states


class MLEDecoderHead(nn.Module):
    def __init__(self, config: MLEConfig, num_layers: int):
        super().__init__()

        self.layer = MLEEncoderLayer(
            config,
            in_features=config.block_patch_size,
            out_features=config.last_hidden_channels,
            stride_sizes=[1 for _ in range(num_layers)],
            num_layers=num_layers,
        )
        self.norm = MLEBatchNorm(config, in_features=config.last_hidden_channels)
        self.conv = nn.Conv2d(
            config.last_hidden_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        pixel_values = self.conv(hidden_states)
        return pixel_values


class MLEDecoder(nn.Module):
    def __init__(
        self,
        config: MLEConfig,
    ):
        super().__init__()

        encoder_output_channels = (
            config.in_channels
            * config.block_patch_size
            * (config.upsample_ratio ** (len(config.num_encoder_layers) - 1))
        )
        upsample_ratio = config.upsample_ratio
        num_decoder_layers = config.num_decoder_layers

        self.layers = nn.ModuleList(
            [
                (
                    MLEDecoderLayer(
                        config,
                        in_features=encoder_output_channels // (upsample_ratio**i),
                        out_features=encoder_output_channels
                        // (upsample_ratio ** (i + 1)),
                        num_layers=num_layers,
                    )
                    if i < len(num_decoder_layers) - 1
                    else MLEDecoderHead(
                        config,
                        num_layers=num_layers,
                    )
                )
                for i, num_layers in enumerate(num_decoder_layers)
            ]
        )

    def forward(
        self,
        last_hidden_states: torch.Tensor,
        encoder_hidden_states: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        hidden_states = last_hidden_states
        num_encoder_hidden_states = len(encoder_hidden_states)  # 5

        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                hidden_states = layer(
                    hidden_states,
                    # 0, 1, 2, 3, 4
                    # ↓  ↓  ↓  ↓  ↓
                    # 8, 7, 6, 5, 5
                    encoder_hidden_states[num_encoder_hidden_states - 2 - i],
                )
            else:
                # decoder head
                hidden_states = layer(hidden_states)

        return hidden_states


class MLEPretrainedModel(PreTrainedModel):
    config_class = MLEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True


class MLEModel(MLEPretrainedModel):
    def __init__(self, config: MLEConfig):
        super().__init__(config)
        self.config = config

        self.encoder = MLEEncoder(config)
        self.decoder = MLEDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        encoder_output, all_hidden_states = self.encoder(pixel_values)
        decoder_output = self.decoder(encoder_output, all_hidden_states)

        return decoder_output


class MLEForAnimeLineExtraction(MLEPretrainedModel):
    def __init__(self, config: MLEConfig):
        super().__init__(config)

        self.model = MLEModel(config)

    def postprocess(self, output_tensor: torch.Tensor, input_shape: tuple[int, int]):
        pixel_values = output_tensor[:, 0, :, :]
        pixel_values = torch.clip(pixel_values, 0, 255)

        pixel_values = pixel_values[:, 0 : input_shape[0], 0 : input_shape[1]]
        return pixel_values

    def forward(
        self, pixel_values: torch.Tensor, return_dict: bool = True
    ) -> tuple[torch.Tensor, ...] | MLEForAnimeLineExtractionOutput:
        # height, width
        input_image_size = (pixel_values.shape[2], pixel_values.shape[3])

        model_output = self.model(pixel_values)

        if not return_dict:
            return (model_output, self.postprocess(model_output, input_image_size))

        else:
            return MLEForAnimeLineExtractionOutput(
                last_hidden_state=model_output,
                pixel_values=self.postprocess(model_output, input_image_size),
            )
