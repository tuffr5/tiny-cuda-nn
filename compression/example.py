# Adapted from tiny-cuda-nn/examples/mlp_learning_an_image.py
# for image compression purpose by
# @Author: Bin Duan <bduan2@hawk.iit.edu>

import argparse
import os
import sys
import time

import commentjson as json
import torch
import torch.nn.functional as F

from bitstream.encode import encode_frame
from bitstream.decode import decode_frame
from models.network import NetwortwithInputEncoding
from utils.common import read_image, write_image
from utils.misc import generate_input_grid


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


class Image(torch.nn.Module):
    @torch.no_grad()
    def __init__(self, filename, device):
        super(Image, self).__init__()
        self.data = torch.from_numpy(read_image(filename)).float().to(device)
        self.data = self.data.permute(2, 0, 1).unsqueeze_(0)
        self.shape_tensor = torch.tensor([self.data.shape[-2], self.data.shape[-1]], device=device).float()

    @torch.no_grad()
    def forward(self, xs, mode='bicubic'):
        xs = (2.0 * xs - 1.0).unsqueeze_(0).unsqueeze_(-2)
        sampled = F.grid_sample(self.data, xs, mode=mode, padding_mode='border', align_corners=True)
        return sampled.squeeze_(0).squeeze_(-1).permute(1, 0)
        

def compute_loss(output, targets):
    relative_l2_error = (output - targets.to(output.dtype))**2 / (output.detach()**2 + 0.01)
    return relative_l2_error.mean()
        

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
    print("===============================================================")
    print("This script is an image compression example using tiny_cuda_nn.")
    print(f"Using PyTorch {torch.__version__} with CUDA {torch.version.cuda}")
    print("===============================================================")

    device = torch.device("cuda")
    args = get_args()

    with open(args.config) as config_file:
        config = json.load(config_file)

    image = Image(args.image, device)
    n_channels, height, width = image.data.shape[1:]
    img_shape = (height, width, n_channels)
    n_pixels = height * width

    model = NetwortwithInputEncoding(n_input_dims=2,
                              n_output_dims=n_channels,
                              encoding_config=config["encoding"],
                              network_config=config["network"]).to(device)
    print(model)

    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    xy = generate_input_grid(img_shape, device)

    path = f"compression/results/reference.jpg"
    print(f"Writing '{path}'... ", end="")
    write_image(path, image(xy).clamp(0.0, 1.0).reshape(img_shape).detach().cpu().numpy())
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
        loss = compute_loss(output, targets)

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
        print(f'Real rate    [kBytes]: {real_rate_byte / 1000:9.3f}')
        print(f'Real rate       [bpp]: {real_rate_bpp :9.3f}')
    
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

    # clean up
    tcnn.free_temporary_memory()