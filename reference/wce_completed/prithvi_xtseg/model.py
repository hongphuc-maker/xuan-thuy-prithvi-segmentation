from __future__ import annotations
import math
from pathlib import Path
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from terratorch.models.necks import Neck
from terratorch.models.encoder_decoder_factory import EncoderDecoderFactory
from terratorch.registry import TERRATORCH_NECK_REGISTRY, BACKBONE_REGISTRY
from terratorch.models.backbones.prithvi_vit import checkpoint_filter_fn_vit, PRETRAINED_BANDS
from xuanthuy_seg.contracts import sha256_file


@TERRATORCH_NECK_REGISTRY.register
class TargetTimeSlice(Neck):
    """Read target-date tokens after joint spatiotemporal attention, before the spatial pyramid.

    ReshapeTokensToImage concatenates (time,embedding) channels. This readout retains
    the last date's embeddings, which have already attended to all input dates.
    This is a project adaptation, not a built-in Prithvi scientific guarantee.
    """
    def __init__(self, channel_list, frames, target_index=-1):
        super().__init__(channel_list)
        self.frames = frames
        self.target_index = target_index % frames
        if any(c % frames for c in channel_list):
            raise ValueError("Temporal channels do not divide into frames")
    def process_channel_list(self, channel_list):
        return [c // self.frames for c in channel_list]
    def forward(self, features, **kwargs):
        return [x.reshape(x.shape[0],self.frames,x.shape[1]//self.frames,*x.shape[-2:])[:,self.target_index]
                for x in features]


class CheckpointBlock(nn.Module):
    def __init__(self, inner):
        super().__init__(); self.inner = inner
    def forward(self, x):
        if self.training and torch.is_grad_enabled():
            return checkpoint(self.inner, x, use_reentrant=False)
        return self.inner(x)


def replace_batch_norm(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            groups = math.gcd(32, child.num_features)
            while child.num_features // groups < 2 and groups > 1:
                groups //= 2
            new = nn.GroupNorm(groups, child.num_features, eps=child.eps, affine=child.affine)
            if child.affine:
                with torch.no_grad():
                    new.weight.copy_(child.weight); new.bias.copy_(child.bias)
            setattr(module,name,new)
        else:
            replace_batch_norm(child)


def build_model(config, pretrained=True, test_override=None):
    m = config["model"]; frames = len(config["active_dates"])
    kwargs = dict(pretrained=False, num_frames=frames,
                  bands=["BLUE","GREEN","RED","NIR_NARROW","SWIR_1","SWIR_2"],
                  out_indices=m["out_indices"], coords_encoding=["time","location"])
    if test_override:
        kwargs.update(test_override)
    backbone = BACKBONE_REGISTRY.build(m["backbone"], **kwargs)
    # v1.2.12 filters forward outputs but leaves the full channel list unchanged.
    backbone.out_channels = [backbone.out_channels[i] for i in kwargs["out_indices"]]
    provenance = {"pretrained": bool(pretrained), "revision":config["model_hf_revision"]}
    if pretrained:
        from huggingface_hub import hf_hub_download
        weights = Path(hf_hub_download("ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL",
                       "Prithvi_EO_V2_300M_TL.pt", revision=config["model_hf_revision"]))
        state = torch.load(weights,map_location="cpu",weights_only=True)
        filtered = checkpoint_filter_fn_vit(state,backbone,PRETRAINED_BANDS,backbone.model_bands)
        # Strict loading: never silently train a partly random backbone.
        backbone.load_state_dict(filtered,strict=True)
        provenance["checkpoint_sha256"] = sha256_file(weights)
        del state, filtered
    if m["gradient_checkpointing"]:
        backbone.blocks = nn.ModuleList([CheckpointBlock(b) for b in backbone.blocks])
    model = EncoderDecoderFactory().build_model(
        task="segmentation",backbone=backbone,decoder=m["decoder"],num_classes=11,
        decoder_channels=m["decoder_channels"], head_dropout=m["head_dropout"],rescale=True,
        necks=[{"name":"ReshapeTokensToImage","effective_time_dim":frames},
               {"name":"TargetTimeSlice","frames":frames,"target_index":frames-1},
               {"name":"LearnedInterpolateToPyramidal"}])
    # A micro-batch of 1 and PPM's 1x1 pooling make BatchNorm invalid in training.
    replace_batch_norm(model)
    model.pretrained_provenance = provenance
    return model


def predict_logits(model, batch):
    out = model(batch["image"],temporal_coords=batch["temporal_coords"],location_coords=batch["location_coords"])
    logits = out.output
    expected = (batch["image"].shape[0],11,*batch["image"].shape[-2:])
    if tuple(logits.shape) != expected:
        raise ValueError(f"Wrong segmentation shape {tuple(logits.shape)}; expected {expected}")
    return logits
