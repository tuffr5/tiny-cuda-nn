#!/usr/bin/env python3

# Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
#       to endorse or promote products derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# @originalfile   mlp_learning_an_image_pytorch.py
# @author Thomas Müller, NVIDIA

# @current: encode.py
# Adapted for image compression purpose by
# @Author: Bin Duan <bduan2@hawk.iit.edu>

import argparse
import commentjson as json
import numpy as np
import os
import sys
import torch
import time
from utils.misc import generate_input_grid, FIXED_POINT_FRACTIONAL_MULT
from models.quant import QuantizableModule
from bitstream.encode import encode_frame
from bitstream.decode import decode_frame

try:
    import tinycudann as tcnn
except ImportError:
    print("This sample requires the tiny-cuda-nn extension for PyTorch.")
    print("You can install it by running:")
    print("============================================================")
    print("tiny-cuda-nn$ cd bindings/torch")
    print("tiny-cuda-nn/bindings/torch$ python setup.py install")
    print("============================================================")
    sys.exit()

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
sys.path.insert(0, SCRIPTS_DIR)

from common import write_image, read_image


class Image(torch.nn.Module):
    def __init__(self, filename, device):
        super(Image, self).__init__()
        self.data = read_image(filename)
        self.shape = self.data.shape
        self.data = torch.from_numpy(self.data).float().to(device)

    def forward(self, xs):
        with torch.no_grad():
            # Bilinearly filtered lookup from the image. Not super fast,
            # but less than ~20% of the overall runtime of this example.
            shape = self.shape

            xs = xs * torch.tensor([shape[1], shape[0]], device=xs.device).float()
            indices = xs.long()
            lerp_weights = xs - indices.float()

            x0 = indices[:, 0].clamp(min=0, max=shape[1] - 1)
            y0 = indices[:, 1].clamp(min=0, max=shape[0] - 1)
            x1 = (x0 + 1).clamp(max=shape[1] - 1)
            y1 = (y0 + 1).clamp(max=shape[0] - 1)

            return (
                self.data[y0, x0] * (1.0 - lerp_weights[:, 0:1]) * (1.0 - lerp_weights[:, 1:2])
                + self.data[y0, x1] * lerp_weights[:, 0:1] * (1.0 - lerp_weights[:, 1:2])
                + self.data[y1, x0] * (1.0 - lerp_weights[:, 0:1]) * lerp_weights[:, 1:2]
                + self.data[y1, x1] * lerp_weights[:, 0:1] * lerp_weights[:, 1:2]
            )

def get_args():
    parser = argparse.ArgumentParser(description="Image Encoding.")

    parser.add_argument("image", nargs="?", default="data/images/kodim04.png", help="Image to match")
    parser.add_argument("config", nargs="?", default="data/config_hash.json", help="JSON config for tiny-cuda-nn",)
    parser.add_argument("n_steps", nargs="?", type=int, default=1001, help="Number of training steps")
    parser.add_argument("bitstream_path", nargs="?", default="compression/results/bitstream.bin", help="path to bitstream file")
    parser.add_argument("result_filename", nargs="?", default="rec", help="recovered image by quantized model")
    parser.add_argument("lmbda", nargs="?", default=0.0001, help="lambda for RD cost")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    start_time = time.time()
    print("================================================================")
    print("This script replicates the behavior of the native CUDA example  ")
    print("mlp_learning_an_image.cu using tiny-cuda-nn's PyTorch extension.")
    print("================================================================")

    print(
        f"Using PyTorch version {torch.__version__} with CUDA {torch.version.cuda}"
    )

    device = torch.device("cuda")
    args = get_args()

    with open(args.config) as config_file:
        config = json.load(config_file)

    image = Image(args.image, device)
    n_channels = image.data.shape[2]

    model = QuantizableModule(n_input_dims=2,
                              n_output_dims=n_channels,
                              encoding_config=config["encoding"],
                              network_config=config["network"]).to(device)
    print(model)

    # Count the number of parameters.param
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")

    # ===================================================================================================
    # The following is equivalent to the above, but slower. Only use "naked" tcnn.Encoding and
    # tcnn.Network when you don't want to combine them. Otherwise, use tcnn.NetworkWithInputEncoding.
    # ===================================================================================================
    # encoding = tcnn.Encoding(n_input_dims=2, encoding_config=config["encoding"])
    # network = tcnn.Network(n_input_dims=encoding.n_output_dims, n_output_dims=n_channels, network_config=config["network"])
    # model = torch.nn.Sequential(encoding, network)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Variables for saving/displaying image results
    resolution = image.data.shape[0:2]
    n_pixels = np.prod(resolution)
    img_shape = resolution + torch.Size([image.data.shape[2]])
    xy = generate_input_grid(img_shape, device)

    path = f"compression/results/reference.jpg"
    print(f"Writing '{path}'... ", end="")
    write_image(path, image(xy).reshape(img_shape).detach().cpu().numpy())
    print("done.")

    prev_time = time.perf_counter()

    batch_size = 2**18
    interval = 10

    print(f"Beginning optimization with {args.n_steps} training steps.")

    try:
        batch = torch.rand([batch_size, 2], device=device, dtype=torch.float32)
        traced_image = torch.jit.trace(image, batch)
    except:
        # If tracing causes an error, fall back to regular execution
        print(f"WARNING: PyTorch JIT trace failed. Performance will be slightly worse than regular.")
        traced_image = image

    for i in range(args.n_steps):
        batch = torch.rand([batch_size, 2], device=device, dtype=torch.float32)
        targets = traced_image(batch)

        output = model(batch)

        relative_l2_error = (output - targets.to(output.dtype))**2 / (output.detach()**2 + 0.01)
        loss = relative_l2_error.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % interval == 0:
            loss_val = loss.item()
            torch.cuda.synchronize()
            elapsed_time = time.perf_counter() - prev_time
            print(
                f"Step#{i}: loss={loss_val} time={int(elapsed_time*1000)}[ms]")

            path = f"compression/results/{i}.jpg"
            print(f"Writing '{path}'... ", end="")
            with torch.no_grad():
                write_image(
                    path,
                    model(xy).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy(),
                )
            print("done.")

            # Ignore the time spent saving the image
            prev_time = time.perf_counter()

            if i > 0 and interval < 1000:
                interval *= 10

    # quantize model
    with torch.no_grad():
        model.eval()

        best_loss = 1e6
        model.save_full_precision_param()

        # Try to find the best quantization step
        for current_q_step in model._POSSIBLE_Q_STEP:
            # Quantize
            quantization_success = model.quantize(current_q_step)

            if not quantization_success:
                continue

            # Measure rate
            rate_bpp = model.measure_laplace_rate() / n_pixels
            relative_l2_error = ((model(xy) - image(xy))**2).mean()
            loss = args.lmbda * rate_bpp + relative_l2_error

            # Store best quantization steps
            if loss < best_loss:
                best_q_step = current_q_step
                best_loss = loss
            
        model.save_q_step(best_q_step)

        assert model.quantize(model._q_step)

        # write to bitstream
        assert encode_frame(model, args.bitstream_path, img_shape, config)

        real_rate_byte = os.path.getsize(args.bitstream_path)
        real_rate_bpp = real_rate_byte * 8 / n_pixels
        print(f'Real rate        [kBytes]: {real_rate_byte / 1000:9.3f}')
        print(f'Real rate           [bpp]: {real_rate_bpp :9.3f}')
    
    print(f'Total encoding time elapsed {time.time() - start_time}')

    # try decoding
    start_time = time.time()
    dec_img = decode_frame(args.bitstream_path, device)
    if args.result_filename:
        print(f"Writing f'compression/results/{args.result_filename}.jpg'... ", end="")
        write_image(
            f'compression/results/{args.result_filename}.jpg',
            dec_img.detach().cpu().numpy())
        print("done.")
    print(f'Total decoding time elapsed {time.time() - start_time}')

    tcnn.free_temporary_memory()
