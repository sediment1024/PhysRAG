import types
from typing import List, Optional
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceConcatInjection(nn.Module):
    def __init__(self, hidden_size: int, adapter_dim: int = 16, seq_len: int = 6750, ref_dim: Optional[int] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.adapter_proj = nn.Linear(adapter_dim, hidden_size)
        self.ref_proj: Optional[nn.Linear] = None
        if ref_dim is not None and ref_dim not in (adapter_dim, hidden_size):
            self.ref_proj = nn.Linear(ref_dim, hidden_size)
        self.seq_proj = nn.Linear(2 * seq_len, seq_len)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states: torch.Tensor, ref_features: torch.Tensor) -> torch.Tensor:
        if ref_features.dtype != hidden_states.dtype:
            ref_features = ref_features.to(hidden_states.dtype)

        batch, seq_hidden, _ = hidden_states.shape
        seq_ref = ref_features.shape[1]

        if ref_features.shape[-1] == hidden_states.shape[-1]:
            ref = ref_features
        elif ref_features.shape[-1] == self.adapter_proj.in_features:
            ref = self.adapter_proj(ref_features)
        elif self.ref_proj is not None and ref_features.shape[-1] == self.ref_proj.in_features:
            ref = self.ref_proj(ref_features)
        else:
            raise RuntimeError(
                f"Unsupported ref_features dim: {ref_features.shape[-1]} "
                f"(hidden_size={self.hidden_size}, adapter_dim={self.adapter_proj.in_features})"
            )

        if seq_ref != seq_hidden:
            ref = ref.permute(0, 2, 1)
            ref = F.interpolate(ref, size=seq_hidden, mode='linear', align_corners=False)
            ref = ref.permute(0, 2, 1)

        cat = torch.cat([hidden_states, ref], dim=1)
        cat_t = cat.permute(0, 2, 1)

        if cat_t.shape[-1] != self.seq_proj.in_features:
            old_weight = self.seq_proj.weight.data
            old_bias = self.seq_proj.bias.data if self.seq_proj.bias is not None else None

            old_weight_2d = old_weight.unsqueeze(0).unsqueeze(0)
            new_weight_2d = F.interpolate(
                old_weight_2d,
                size=(seq_hidden, cat_t.shape[-1]),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0).squeeze(0)

            if old_bias is not None:
                new_bias = F.interpolate(
                    old_bias.unsqueeze(0).unsqueeze(0),
                    size=(1, seq_hidden),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0).squeeze(0)
            else:
                new_bias = None

            self.seq_proj = nn.Linear(
                cat_t.shape[-1],
                seq_hidden,
                bias=(new_bias is not None),
            ).to(device=old_weight.device, dtype=old_weight.dtype)
            self.seq_proj.weight.data = new_weight_2d
            if new_bias is not None:
                self.seq_proj.bias.data = new_bias

        out_t = self.seq_proj(cat_t)
        out = out_t.permute(0, 2, 1)
        return hidden_states + self.alpha * out


class BlockWrapper(nn.Module):
    def __init__(self, original_block: nn.Module, injection_module: nn.Module):
        super().__init__()
        self.original_block = original_block
        self.injection = injection_module
        self.current_ref: Optional[torch.Tensor] = None
        self.model_ref: Optional[weakref.ReferenceType] = None
        self.query_mode: str = "learned"
        self.use_direct_tokens: bool = False

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        ref = self.current_ref
        if self.query_mode == "dit":
            model_ref = self.model_ref
            model = model_ref() if model_ref is not None else None
            if model is not None:
                ref_feat = getattr(model, "current_ref_features", None)
                if ref_feat is not None:
                    adapter = getattr(model, "physical_adapter", None)
                    if adapter is not None and not self.use_direct_tokens:
                        device = hidden_states.device
                        dtype = hidden_states.dtype
                        ref_feat = ref_feat.to(device=device, dtype=dtype)
                        if ref_feat.shape[0] == 1 and hidden_states.shape[0] > 1:
                            ref_feat = ref_feat.expand(hidden_states.shape[0], -1, -1)
                        adapter_out = adapter(
                            ref_features=ref_feat,
                            target_shape=getattr(model, "current_target_shape", None),
                            query_override=hidden_states,
                        )
                        if adapter_out.dim() == 3:
                            ref = adapter_out
        if ref is not None:
            hidden_states = self.injection(hidden_states, ref)
        return self.original_block(hidden_states, *args, **kwargs)


class HeadWrapper(nn.Module):
    def __init__(self, original_head: nn.Module, injection_module: nn.Module):
        super().__init__()
        self.original_head = original_head
        self.injection = injection_module
        self.current_ref: Optional[torch.Tensor] = None
        self.model_ref: Optional[weakref.ReferenceType] = None
        self.query_mode: str = "learned"
        self.use_direct_tokens: bool = False

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        ref = self.current_ref
        if self.query_mode == "dit":
            model_ref = self.model_ref
            model = model_ref() if model_ref is not None else None
            if model is not None:
                ref_feat = getattr(model, "current_ref_features", None)
                if ref_feat is not None:
                    adapter = getattr(model, "physical_adapter", None)
                    if adapter is not None and not self.use_direct_tokens:
                        device = hidden_states.device
                        dtype = hidden_states.dtype
                        ref_feat = ref_feat.to(device=device, dtype=dtype)
                        if ref_feat.shape[0] == 1 and hidden_states.shape[0] > 1:
                            ref_feat = ref_feat.expand(hidden_states.shape[0], -1, -1)
                        adapter_out = adapter(
                            ref_features=ref_feat,
                            target_shape=getattr(model, "current_target_shape", None),
                            query_override=hidden_states,
                        )
                        if adapter_out.dim() == 3:
                            ref = adapter_out
        if ref is not None:
            hidden_states = self.injection(hidden_states, ref)
        return self.original_head(hidden_states, *args, **kwargs)

class PhysicalInjectionManager:
    def __init__(
        self,
        model: nn.Module,
        physical_adapter: nn.Module,
        injection_layers: List[int],
        hidden_size: int,
        adapter_dim: int = 16,
        seq_len: int = 6750,
        patch_size: tuple = (1, 2, 2),
        dtype: torch.dtype = torch.bfloat16,
        use_direct_tokens: bool = False,
        injection_position: str = "layers",
        query_mode: str = "learned",
    ):
        self.model = model
        self.physical_adapter = physical_adapter
        self.injection_layers = injection_layers
        self.hidden_size = hidden_size
        self.adapter_dim = adapter_dim
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.dtype = dtype
        self.use_direct_tokens = use_direct_tokens
        self.injection_position = injection_position
        self.query_mode = query_mode
        self.direct_ref_dim = 768 if self.use_direct_tokens else None

        if self.physical_adapter is not None:
            self.model.physical_adapter = self.physical_adapter
        self.model._injection_layers = list(self.injection_layers)
        self.model._use_direct_tokens = self.use_direct_tokens
        self.model._injection_position = self.injection_position
        self.model._physical_query_mode = self.query_mode

        self.original_forward = self.model.forward

        self._inject_layers()
        self._setup_forward()

    def _inject_layers(self):
        device = self.model.patch_embedding.weight.device
        injection_kwargs = dict(
            hidden_size=self.hidden_size,
            adapter_dim=self.adapter_dim,
            seq_len=self.seq_len,
            ref_dim=self.direct_ref_dim,
        )
        if self.injection_position == "post":
            original_head = self.model.head
            injection = SequenceConcatInjection(**injection_kwargs).to(device=device, dtype=self.dtype)
            wrapped_head = HeadWrapper(original_head, injection)
            wrapped_head.model_ref = weakref.ref(self.model)
            wrapped_head.query_mode = self.query_mode
            wrapped_head.use_direct_tokens = self.use_direct_tokens
            self.model.head = wrapped_head
            return

        for idx in self.injection_layers:
            original_block = self.model.blocks[idx]
            injection = SequenceConcatInjection(**injection_kwargs).to(device=device, dtype=self.dtype)

            wrapped_block = BlockWrapper(original_block, injection)
            wrapped_block.model_ref = weakref.ref(self.model)
            wrapped_block.query_mode = self.query_mode
            wrapped_block.use_direct_tokens = self.use_direct_tokens
            self.model.blocks[idx] = wrapped_block

    def _setup_forward(self):
        original_forward = self.original_forward

        def new_forward(self, x, t, context, seq_len, y=None):
            ref_feat = getattr(self, "current_ref_features", None)
            target_shape_tensor = getattr(self, "current_target_shape", None)

            if ref_feat is not None and target_shape_tensor is not None:
                device = self.patch_embedding.weight.device
                dtype = self.patch_embedding.weight.dtype

                ref_feat = ref_feat.to(device=device, dtype=dtype)
                use_direct = getattr(self, "_use_direct_tokens", False)
                injection_position = getattr(self, "_injection_position", "layers")
                query_mode = getattr(self, "_physical_query_mode", "learned")
                if use_direct:
                    batch_size = len(x)
                    if ref_feat.dim() == 2:
                        ref_feat = ref_feat.unsqueeze(0)
                    if ref_feat.shape[0] == 1 and batch_size > 1:
                        ref_feat = ref_feat.expand(batch_size, -1, -1)
                    adapter_flat = ref_feat
                elif query_mode != "dit":
                    self.physical_adapter.to(device=device, dtype=dtype)

                    batch_size = len(x)
                    if ref_feat.shape[0] == 1 and batch_size > 1:
                        ref_feat = ref_feat.expand(batch_size, -1, -1)

                    adapter_out = self.physical_adapter(
                        ref_features=ref_feat,
                        target_shape=target_shape_tensor,
                    )

                    # If adapter returns tokens, use them directly; otherwise pool 3D features.
                    if adapter_out.dim() == 3:
                        adapter_flat = adapter_out
                    else:
                        h_even = adapter_out.size(3) - (adapter_out.size(3) % self.patch_size[1])
                        w_even = adapter_out.size(4) - (adapter_out.size(4) % self.patch_size[2])
                        if h_even != adapter_out.size(3) or w_even != adapter_out.size(4):
                            adapter_out = adapter_out[:, :, :, :h_even, :w_even]

                        pool_kernel = (1, self.patch_size[1], self.patch_size[2])
                        pooled_adapter = F.avg_pool3d(
                            adapter_out,
                            kernel_size=pool_kernel,
                            stride=pool_kernel,
                        )

                        bp, cp, fp, hp, wp = pooled_adapter.shape
                        adapter_flat = pooled_adapter.permute(0, 2, 3, 4, 1).reshape(bp, -1, cp)

                if query_mode != "dit":
                    if injection_position == "post":
                        if hasattr(self, "head") and hasattr(self.head, "current_ref"):
                            self.head.current_ref = adapter_flat
                    else:
                        injection_layers = getattr(self, "_injection_layers", [])
                        for idx in injection_layers:
                            self.blocks[idx].current_ref = adapter_flat
                else:
                    if injection_position == "post":
                        if hasattr(self, "head") and hasattr(self.head, "current_ref"):
                            self.head.current_ref = None
                    else:
                        injection_layers = getattr(self, "_injection_layers", [])
                        for idx in injection_layers:
                            self.blocks[idx].current_ref = None

            return original_forward(
                x,
                t,
                context,
                seq_len,
                y=y,
            )

        self.model.forward = types.MethodType(new_forward, self.model)

    def set_ref_features(self, ref_features: torch.Tensor, target_shape: torch.Size):
        self.model.current_ref_features = ref_features
        self.model.current_target_shape = target_shape
