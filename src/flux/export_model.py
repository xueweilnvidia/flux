import os
import re
import time
from dataclasses import dataclass
from glob import iglob

import torch
from einops import rearrange
from fire import Fire
from PIL import ExifTags, Image

from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import (configs, embed_watermark, load_ae, load_clip,
                       load_flow_model, load_t5)
from transformers import pipeline

device = "cuda"
model = load_flow_model("flux-dev", device=device)


H = 512 / 8
W = 512 / 8

img = torch.randn(1, int(H*W/4), 64, dtype=torch.bfloat16).to(device)
img_ids = torch.randn(1, int(H*W/4), 3, dtype=torch.float32).to(device)
txt = torch.randn(1, 512, 4096, dtype=torch.bfloat16).to(device)
txt_ids = torch.randn(1, 512, 3, dtype=torch.float32).to(device)
y = torch.randn(1, 768, dtype=torch.bfloat16).to(device)
t_vec = torch.randn(1, dtype=torch.bfloat16).to(device)
guidance = torch.randn(1, dtype=torch.bfloat16).to(device)



torch.onnx.export(
    model, 
    (img, img_ids, txt, txt_ids,  t_vec, y, guidance), 
    "./onnx_model/model.onnx",
    export_params=True,  # Store the trained parameter weights inside the model file
    opset_version=18,    # ONNX version to export to
    do_constant_folding=True,  # Whether to execute constant folding for optimization
    input_names=['img', 'img_ids', 'txt', 'txt_ids',  't_vec', 'y', 'guidance'],
    output_names=['output'],  # Change 'output' to match your model's output name
    dynamic_axes={
        'img': {0: 'batch_size', 1: 'sequence'},
        'img_ids': {0: 'batch_size', 1: 'sequence'},
        'txt': {0: 'batch_size'},
        'txt_ids': {0: 'batch_size'},
        't_vec': {0: 'batch_size'},
        'y': {0: 'batch_size'},
        'guidance': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
)