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
import tensorrt as trt
from cuda import cudart

device = "cuda"
model = load_flow_model("flux-dev", device=device)


H = 1024 / 8
W = 1024 / 8

img = torch.randn(1, int(H*W/4), 64, dtype=torch.bfloat16).to(device).contiguous()
img_ids = torch.randn(1, int(H*W/4), 3, dtype=torch.float32).to(device).contiguous()
txt = torch.randn(1, 512, 4096, dtype=torch.bfloat16).to(device).contiguous()
txt_ids = torch.randn(1, 512, 3, dtype=torch.float32).to(device).contiguous()
y = torch.randn(1, 768, dtype=torch.bfloat16).to(device).contiguous()
t_vec = torch.randn(1, dtype=torch.bfloat16).to(device).contiguous()
guidance = torch.randn(1, dtype=torch.bfloat16).to(device).contiguous()


out = model(img, img_ids, txt, txt_ids,  t_vec, y, guidance)
print("out")
print(out)

output = torch.zeros(1, int(H*W/4), 64, dtype=torch.bfloat16).to(device).contiguous()


trt_logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(trt_logger, '')

with open("test.engine", "rb") as f:
    engine_str = f.read()
engine = trt.Runtime(trt_logger).deserialize_cuda_engine(engine_str)
context = engine.create_execution_context()



tensor_list = [img, img_ids, txt, txt_ids,  t_vec, y, guidance, output]

# context.set_input_shape("/down_blocks.1/attentions.0/transformer_blocks.0/norm1/LayerNormalization_output_0", [1, 640, 640])

for i in range(7):
    name = engine.get_tensor_name(i)
    print(name)
    context.set_input_shape(name, tensor_list[i].shape)
    context.set_tensor_address(name, tensor_list[i].data_ptr())

name = engine.get_tensor_name(7)
context.set_tensor_address(name, tensor_list[7].data_ptr())

_, stream = cudart.cudaStreamCreate()
context.execute_async_v3(stream)
cudart.cudaStreamSynchronize(stream)
# with nvtx.annotate(message="trt_fp8", color="green"):
#     context.execute_async_v3(stream)
#     cudart.cudaStreamSynchronize(stream)
print(output)