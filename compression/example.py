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
from models.network import QuantizableNetworkWithInputEncoding
from models.trainer import Trainer
from utils.common import read_image, write_image
from utils.misc import generate_input_grid
import compressai.utils.bench.codecs as bench


try:
    import tinycudann as tcnn

    # Define the callback function with a logging threshold
    def log_callback(severity, message):
        print(f"[{severity}] {message}")
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
        sampled = F.grid_sample(self.data, xs, mode=mode, padding_mode='reflection', align_corners=False)
        return sampled.squeeze_(0).squeeze_(-1).permute(1, 0)
        

def get_args():
    parser = argparse.ArgumentParser(description="Image compression example.")

    parser.add_argument("image_str", nargs="?", default="data/images/kodim04.png", help="Image to match")
    parser.add_argument("config", nargs="?", default="data/config_hash.json", help="JSON config for tiny-cuda-nn",)
    parser.add_argument("bitstream_path", nargs="?", default="compression/results/bitstream.bin", help="path to bitstream file")
    parser.add_argument("result_filename", nargs="?", default="rec", help="recovered image by quantized model")
    
    # model related args
    parser.add_argument("batch_size", nargs="?", default=2**18, help="batch size")
    parser.add_argument("lr", nargs="?", default=0.01, help="learning rate")
    parser.add_argument("n_steps", nargs="?", default=1000, help="number of steps for on-the-fly-training")
    parser.add_argument("device", nargs="?", default="cuda", help="device to use")
    parser.add_argument("n_pixels", nargs="?", default=512*512, help="number of pixels in the image")
    parser.add_argument("lmbda", nargs="?", default=0.002, help="lambda for RD cost")

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

    # Set the log callback
    # tcnn.modules._C.set_log_callback(log_callback)

    with open(args.config) as config_file:
        config = json.load(config_file)

    image = Image(args.image_str, device)
    n_channels, height, width = image.data.shape[1:]
    img_shape = (height, width, n_channels)
    
    args.device = device
    args.n_pixels = height * width

    model = QuantizableNetworkWithInputEncoding(
        n_input_dims=2, 
        n_output_dims=n_channels,
        encoding_config=config["encoding"], 
        network_config=config["network"]).to(device)
    
    print(model)

    trainer = Trainer(model, image, **vars(args))
    trainer.train()

    # write to bitstream
    assert encode_frame(model, args.bitstream_path, img_shape, config)

    real_rate_byte = os.path.getsize(args.bitstream_path)
    real_rate_bpp = real_rate_byte * 8 / args.n_pixels
    print(f'Real rate    [kBytes]: {real_rate_byte / 1000:9.3f}')
    print(f'Real rate       [bpp]: {real_rate_bpp :9.3f}')
    print(f'Total encoding time elapsed {time.time() - start_time} s')

    with torch.no_grad():
        model.to(device)
        image_array = image.data.squeeze(0).permute(1, 2, 0).cpu().numpy()
        xy = generate_input_grid(img_shape, device)
        model.set_param(model.get_full_precision_param())
        path = f"compression/results/enc_fp.jpg"
        print(f"Writing '{path}'... ", end="")
        rec_image = model(xy).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy()
        write_image(path, rec_image)

        print(f"Full-precison Model: {bench.compute_metrics((image_array* 255).astype('uint8'), (rec_image * 255).astype('uint8'), metrics=['psnr', 'ms-ssim'])}")
        model.dequantize()
        path = f"compression/results/enc_quant.jpg"
        print(f"Writing '{path}'... ", end="")
        rec_image = model(xy).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy()
        write_image(path, rec_image)
        print(f"Quantized Model   : {bench.compute_metrics((image_array* 255).astype('uint8'), (rec_image * 255).astype('uint8'), metrics=['psnr', 'ms-ssim'])}")
    print("done.")
    # clean up
    tcnn.free_temporary_memory()

    # try decoding
    start_time = time.time()
    dec_img = decode_frame(args.bitstream_path, device)
    if args.result_filename:
        print(f"Writing f'compression/results/{args.result_filename}.jpg'... ", end="")
        write_image(
            f'compression/results/{args.result_filename}.jpg',
            dec_img.detach().cpu().numpy())
        print("done.")
    print(f'Total decoding time elapsed {time.time() - start_time} s')

    # clean up
    tcnn.free_temporary_memory()