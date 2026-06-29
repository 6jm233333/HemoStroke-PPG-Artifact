from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "BasicBlock1D",
    "ResNetBasicBlock",   # backward-compatible alias
    "ResNet1D",
    "resnet1d18",
]


class BasicBlock1D(nn.Module):
    """
    Standard 1D residual basic block.

    Layout:
        Conv1d -> BN -> ReLU -> Conv1d -> BN -> Add shortcut -> ReLU
    """

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = norm_layer(out_channels)

        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                norm_layer(out_channels * self.expansion),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


# Backward-compatible alias matching your previous code naming
ResNetBasicBlock = BasicBlock1D


class ResNet1D(nn.Module):
    """
    1D ResNet classifier for multivariate sequential inputs.

    Expected input formats:
        - [B, T, C]  (default, recommended for your current dataset)
        - [B, C, T]

    Typical use in this project:
        x.shape = [batch_size, 500, 17]
        model = ResNet1D(input_dim=17, output_dim=2)

    Parameters
    ----------
    input_dim:
        Number of input channels / features per timestep.
    output_dim:
        Number of output classes.
    layers:
        Number of residual blocks in each stage.
        Default (2, 2, 2, 2) corresponds to a ResNet-18 style backbone.
    base_channels:
        Number of channels in the stem and the first stage.
    block:
        Residual block class.
    input_layout:
        One of:
            - "btc": input is [B, T, C]
            - "bct": input is [B, C, T]
            - "auto": infer from shape
    stem_kernel_size:
        Kernel size of the first convolution.
    stem_stride:
        Stride of the first convolution.
    use_maxpool:
        Whether to use the stem max-pooling layer.
    dropout:
        Optional dropout before the final classifier.
    zero_init_residual:
        If True, initialize the second BN in each residual block to zero.
    """

    def __init__(
        self,
        input_dim: int = 17,
        output_dim: int = 2,
        *,
        layers: Sequence[int] = (2, 2, 2, 2),
        base_channels: int = 64,
        block: Type[BasicBlock1D] = BasicBlock1D,
        input_layout: str = "btc",
        stem_kernel_size: int = 7,
        stem_stride: int = 2,
        use_maxpool: bool = True,
        dropout: float = 0.0,
        zero_init_residual: bool = False,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        if len(layers) != 4:
            raise ValueError(f"'layers' must have length 4, got {layers}")

        if input_layout not in {"btc", "bct", "auto"}:
            raise ValueError(f"Unsupported input_layout='{input_layout}'")

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.input_layout = input_layout
        self.base_channels = int(base_channels)
        self.block = block
        self.layers_cfg = tuple(int(x) for x in layers)
        self.dropout_p = float(dropout)

        self.in_channels = self.base_channels

        stem_padding = stem_kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=self.base_channels,
            kernel_size=stem_kernel_size,
            stride=stem_stride,
            padding=stem_padding,
            bias=False,
        )
        self.bn1 = norm_layer(self.base_channels)
        self.relu = nn.ReLU(inplace=True)

        if use_maxpool:
            self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.Identity()

        self.layer1 = self._make_layer(
            out_channels=self.base_channels,
            blocks=self.layers_cfg[0],
            stride=1,
            block=block,
            norm_layer=norm_layer,
        )
        self.layer2 = self._make_layer(
            out_channels=self.base_channels * 2,
            blocks=self.layers_cfg[1],
            stride=2,
            block=block,
            norm_layer=norm_layer,
        )
        self.layer3 = self._make_layer(
            out_channels=self.base_channels * 4,
            blocks=self.layers_cfg[2],
            stride=2,
            block=block,
            norm_layer=norm_layer,
        )
        self.layer4 = self._make_layer(
            out_channels=self.base_channels * 8,
            blocks=self.layers_cfg[3],
            stride=2,
            block=block,
            norm_layer=norm_layer,
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        self.fc = nn.Linear(self.base_channels * 8 * block.expansion, self.output_dim)

        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock1D):
                    nn.init.constant_(m.bn2.weight, 0.0)

    def _make_layer(
        self,
        *,
        out_channels: int,
        blocks: int,
        stride: int,
        block: Type[BasicBlock1D],
        norm_layer: Callable[[int], nn.Module],
    ) -> nn.Sequential:
        layers: List[nn.Module] = []

        layers.append(
            block(
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=stride,
                norm_layer=norm_layer,
            )
        )
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=1,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0.0)

    def _to_bct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert input to [B, C, T] for Conv1d.
        """
        if x.ndim != 3:
            raise ValueError(
                f"ResNet1D expects a 3D tensor, got shape {tuple(x.shape)}"
            )

        if self.input_layout == "bct":
            if x.shape[1] != self.input_dim:
                raise ValueError(
                    f"input_layout='bct' expects x.shape[1] == input_dim ({self.input_dim}), "
                    f"got shape {tuple(x.shape)}"
                )
            return x

        if self.input_layout == "btc":
            if x.shape[2] != self.input_dim:
                raise ValueError(
                    f"input_layout='btc' expects x.shape[2] == input_dim ({self.input_dim}), "
                    f"got shape {tuple(x.shape)}"
                )
            return x.transpose(1, 2)

        # auto mode
        if x.shape[2] == self.input_dim and x.shape[1] != self.input_dim:
            return x.transpose(1, 2)

        if x.shape[1] == self.input_dim and x.shape[2] != self.input_dim:
            return x

        if x.shape[1] == self.input_dim and x.shape[2] == self.input_dim:
            raise ValueError(
                "Ambiguous input shape in auto mode: both channel and time dimensions "
                f"match input_dim={self.input_dim}. Please set input_layout explicitly."
            )

        raise ValueError(
            f"Could not infer input layout for shape {tuple(x.shape)} with input_dim={self.input_dim}. "
            "Expected either [B, T, C] or [B, C, T]."
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the pooled backbone feature vector before the classifier.
        Output shape: [B, feature_dim]
        """
        x = self._to_bct(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Parameters
        ----------
        x:
            Input tensor, usually [B, 500, 17] in this project.
        return_features:
            If True, also return the pooled backbone representation.

        Returns
        -------
        logits:
            Shape [B, output_dim]
        optionally:
            features of shape [B, backbone_dim]
        """
        feats = self.forward_features(x)
        feats = self.dropout(feats)
        logits = self.fc(feats)

        if return_features:
            return logits, feats
        return logits

    @property
    def feature_dim(self) -> int:
        return self.base_channels * 8 * self.block.expansion


def resnet1d18(
    input_dim: int = 17,
    output_dim: int = 2,
    **kwargs,
) -> ResNet1D:
    """
    Convenience factory for the default ResNet-18-style 1D model.
    """
    return ResNet1D(
        input_dim=input_dim,
        output_dim=output_dim,
        layers=(2, 2, 2, 2),
        **kwargs,
    )
