# Header INFO FOR COMPRESSED BITSTREAM
# Adapted from Cool-Chic <https://github.com/Orange-OpenSource/Cool-Chic>
# @Author: Bin Duan <bduan2@hawk.iit.edu>


"""
Bitstream structure
-------------------

Header for the image compressed bitstream:
------------------------------------

    ? ======================== FRAME HEADER ======================== ?
    [Number of bytes used for the header]           2 bytes
    [Image height]                                  2 bytes
    [Image width]                                   2 bytes
    [Number of Image channels]                      1 byte
    [Number of Context points]                      1 byte
    [Number of bytes used for encoding configs]     1 byte
    [Number of bytes used for network configs]      1 byte
    [Number of bytes used for arm configs]          1 byte
    
    # ======================== ARM STUFF =========================== #
    [AC_MAX_VAL]                                    2 bytes
    [scale]                                         4 bytes (float)
    [Number of bytes used for network]              2 bytes 
    # ======================== NETWORK STUFF ======================= #
    [AC_MAX_VAL]                                    2 bytes
    [scale]                                         4 bytes (float)
    [Number of bytes used for network]              2 bytes 
    # ======================== ENCODING STUFF ====================== #
    [AC_MAX_VAL]                                    2 bytes
    [Number of bytes used for encoding]             4 bytes
    ? ======================== FRAME HEADER ======================== ?

"""


import os
import struct
from typing import Tuple, TypedDict
from utils.misc import (
    _ENCODING_TYPE,
    _GRID_TYPE,
    _INTERPOLATION_TYPE,
    _ACTIVATION_TYPE,
    _NETWORK_TYPE,
    get_num_layers,
    get_num_levels
)


def write_encoding_config(config: dict):
    byte_to_write = b""
    n_bytes_encoding = 0
    otype = config["otype"]

    if otype in _ENCODING_TYPE:
        byte_to_write += _ENCODING_TYPE.index(otype).to_bytes(1, byteorder="big", signed=False)
        n_bytes_encoding += 1
    else:
        raise RuntimeError(f"No such {otype} supported")

    if otype == "Identity":
        byte_to_write += struct.pack("f", config["scale"])
        byte_to_write += struct.pack("f", config["offset"])
        n_bytes_encoding += 8
    elif otype == "Frequency":
        byte_to_write += config["n_frequencies"].to_bytes(1, byteorder="big", signed=False)
        n_bytes_encoding += 1
    elif otype == "OneBlob":
        byte_to_write += config["n_bins"].to_bytes(1, byteorder="big", signed=False)
        n_bytes_encoding += 1
    elif otype == "SphericalHarmonics":
        byte_to_write += config["degree"].to_bytes(1, byteorder="big", signed=False)
        n_bytes_encoding += 1
    elif otype == "TriangleWave":
        byte_to_write += config["n_frequencies"].to_bytes(1, byteorder="big", signed=False)
        n_bytes_encoding += 1
    elif otype == "Grid":
        byte_to_write += _GRID_TYPE.index(config["type"]).to_bytes(1, byteorder="big", signed=False)
        byte_to_write += config["n_levels"].to_bytes(1, byteorder="big", signed=False)
        byte_to_write += config["n_features_per_level"].to_bytes(1, byteorder="big", signed=False)
        byte_to_write += config["log2_hashmap_size"].to_bytes(1, byteorder="big", signed=False)
        byte_to_write += config["base_resolution"].to_bytes(1, byteorder="big", signed=False)
        byte_to_write += struct.pack("f", config["per_level_scale"])
        byte_to_write += _INTERPOLATION_TYPE.index(config["interpolation"]).to_bytes(1, byteorder="big", signed=False)
        n_bytes_encoding += 10

    return n_bytes_encoding, byte_to_write


def read_encoding_config(bitstream: bytes) -> dict:
    config = {}
    ptr = 0

    otype = _ENCODING_TYPE[int.from_bytes(bitstream[ptr:ptr + 1], byteorder='big', signed=False)]
    ptr += 1
    config["otype"] = otype

    if otype == "Identity":
        config["scale"] = struct.unpack_from("f", bitstream, ptr)[0]
        config["offset"] = struct.unpack_from("f", bitstream, ptr + 4)[0]
        ptr += 8
    elif otype == "Frequency":
        config["n_frequencies"] = int.from_bytes(bitstream[ptr:ptr + 1], byteorder='big', signed=False)
        ptr += 1
    elif otype == "OneBlob":
        config["n_bins"] = int.from_bytes(bitstream[ptr:ptr + 1], byteorder='big', signed=False)
        ptr += 1
    elif otype == "SphericalHarmonics":
        config["degree"] = int.from_bytes(bitstream[ptr:ptr + 1], byteorder='big', signed=False)
        ptr += 1
    elif otype == "TriangleWave":
        config["n_frequencies"] = int.from_bytes(bitstream[ptr:ptr + 1], byteorder='big', signed=False)
        ptr += 1
    elif otype == "Grid":
        config["type"] = _GRID_TYPE[int.from_bytes(bitstream[ptr:ptr + 1], byteorder='big', signed=False)]
        config["n_levels"] = int.from_bytes(bitstream[ptr + 1:ptr + 2], byteorder='big', signed=False)
        config["n_features_per_level"] = int.from_bytes(bitstream[ptr + 2:ptr + 3], byteorder='big', signed=False)
        config["log2_hashmap_size"] = int.from_bytes(bitstream[ptr + 3:ptr + 4], byteorder='big', signed=False)
        config["base_resolution"] = int.from_bytes(bitstream[ptr + 4:ptr + 5], byteorder='big', signed=False)
        config["per_level_scale"] = struct.unpack_from("f", bitstream, ptr + 5)[0]
        config["interpolation"] = _INTERPOLATION_TYPE[int.from_bytes(bitstream[ptr + 9: ptr + 10], byteorder='big', signed=False)]
        ptr += 10

    return config


def write_network_config(config: dict):
    byte_to_write = b""
    n_bytes_network = 0
    otype = config["otype"]

    if otype in _NETWORK_TYPE:
        byte_to_write += _NETWORK_TYPE.index(otype).to_bytes(1, byteorder="big", signed=False)
        byte_to_write += _ACTIVATION_TYPE.index(config["activation"]).to_bytes(1, byteorder="big", signed=False)
        byte_to_write += _ACTIVATION_TYPE.index(config["output_activation"]).to_bytes(1, byteorder="big", signed=False)
        byte_to_write += config["n_neurons"].to_bytes(1, byteorder="big", signed=False)
        byte_to_write += config["n_hidden_layers"].to_bytes(1, byteorder="big", signed=False)
        n_bytes_network += 5
    else:
        # Handling for other cases (if needed)
        raise RuntimeError(f"No such {otype} supported")

    return n_bytes_network, byte_to_write


def read_network_config(bitstream: bytes) -> dict:
    config = {}

    ptr = 0
    otype = _NETWORK_TYPE[int.from_bytes(bitstream[ptr:ptr+1], byteorder='big', signed=False)]
    ptr += 1

    config["otype"] = otype

    if otype in _NETWORK_TYPE:
        config["activation"] = _ACTIVATION_TYPE[int.from_bytes(bitstream[ptr:ptr+1], byteorder='big', signed=False)]
        config["output_activation"] = _ACTIVATION_TYPE[int.from_bytes(bitstream[ptr+1:ptr+2], byteorder='big', signed=False)]
        config["n_neurons"] = int.from_bytes(bitstream[ptr + 2:ptr + 3], byteorder='big', signed=False)
        config["n_hidden_layers"] = int.from_bytes(bitstream[ptr + 3:ptr + 4], byteorder='big', signed=False)
        ptr += 4
    else:
        # Handling for other cases (if needed)
        raise RuntimeError(f"No such {otype} supported")

    return config


class FrameHeader(TypedDict):
    n_bytes_header: int  # Number of bytes for the header
    img_size: Tuple[int, int, int]  # Format: (HWC)
    n_contexts: int  # Number of context points
    encoding_configs: dict  # _ENCODING_CONFIG
    network_configs: dict  # _NETWORK_CONFIG
    arm_configs: dict  # _ARM_CONFIG
    acmax_scale_and_n_bytes: list[Tuple]  # List of (acmax, scale, bytes)


def write_frame_header(
    header_path: str,
    img_size: Tuple[int, int, int],
    config: dict,
    acmax_scale_and_n_bytes: list[Tuple]
):
    """Write a frame header to a a file located at <header_path>.
    The structure of the header is described above.
    """
    n_bytes_header = 0
    n_bytes_header += 2  # Number of bytes header
    n_bytes_header += 2  # Image height
    n_bytes_header += 2  # Image width
    n_bytes_header += 1  # Image channel
    n_bytes_header += 1  # Number of context points

    n_bytes_header += 1  # Number of bytes of arm configs
    n_bytes_header += 1  # Number of bytes of network configs
    n_bytes_header += 1  # Number of bytes of encoding configs

    # arm stuff
    n_bytes_arm_config, tmp_byte_arm = write_network_config(config["arm"])
    n_bytes_header += n_bytes_arm_config

    # network stuff
    n_bytes_network_config, tmp_byte_network = write_network_config(config["network"])
    n_bytes_header += n_bytes_network_config

    # encoding stuff
    n_bytes_encoding_config, tmp_byte_encoding = write_encoding_config(config["encoding"])
    n_bytes_header += n_bytes_encoding_config

    n_bytes_header += 8 + 8 + 6 # 8 for arm, 8 for network, 6 for encoding

    byte_to_write = b""
    byte_to_write += n_bytes_header.to_bytes(2, byteorder="big", signed=False)
    byte_to_write += img_size[0].to_bytes(2, byteorder="big", signed=False)
    byte_to_write += img_size[1].to_bytes(2, byteorder="big", signed=False)
    byte_to_write += img_size[2].to_bytes(1, byteorder="big", signed=False)

    n_contexts = config["arm"]["n_contexts"]
    byte_to_write += n_contexts.to_bytes(1, byteorder="big", signed=False)

    byte_to_write += n_bytes_arm_config.to_bytes(1, byteorder="big", signed=False)
    byte_to_write += tmp_byte_arm

    byte_to_write += n_bytes_network_config.to_bytes(1, byteorder="big", signed=False)
    byte_to_write += tmp_byte_network

    byte_to_write += n_bytes_encoding_config.to_bytes(1, byteorder="big", signed=False)
    byte_to_write += tmp_byte_encoding

    for i, (acmax_i, scale_i, n_bytes_i) in enumerate(acmax_scale_and_n_bytes):
        byte_to_write += acmax_i.to_bytes(2, byteorder="big", signed=False)
        if i < 2: # net and arm
            byte_to_write += struct.pack("f", scale_i)
            byte_to_write += n_bytes_i.to_bytes(2, byteorder="big", signed=False)
        else: # encoding
            byte_to_write += n_bytes_i.to_bytes(4, byteorder="big", signed=False)

    with open(header_path, "wb") as fout:
        fout.write(byte_to_write)

    if n_bytes_header != os.path.getsize(header_path):
        print("Invalid number of bytes in header!")
        print("expected", n_bytes_header)
        print("got", os.path.getsize(header_path))
        return False

    print('\nContent of the frame header:')
    print('------------------------------')
    print(f'"n_bytes_header": {n_bytes_header}')
    print(f'"img_size": {img_size}')
    print(f'"n_contexts": {n_contexts}')
    print(f'"acmax_scale_and_n_bytes": {acmax_scale_and_n_bytes}')
    print('         ------------------------')

    return True


def read_frame_header(header_bytes: bytes) -> FrameHeader:
    header = FrameHeader()

    # Parse the header bytes according to the format
    ptr = 0
    n_bytes_header = int.from_bytes(header_bytes[ptr:ptr + 2], byteorder='big', signed=False)
    ptr += 2

    img_size = (
        int.from_bytes(header_bytes[ptr:ptr + 2], byteorder='big', signed=False),
        int.from_bytes(header_bytes[ptr+2:ptr + 4], byteorder='big', signed=False),
        int.from_bytes(header_bytes[ptr+4:ptr + 5], byteorder='big', signed=False)
    )
    ptr += 5

    n_contexts = int.from_bytes(header_bytes[ptr:ptr + 1], byteorder='big', signed=False)
    ptr += 1

    # Read arm configuration
    n_bytes_arm_config = int.from_bytes(header_bytes[ptr:ptr + 1], byteorder='big', signed=False)
    ptr += 1
    arm_configs = read_network_config(header_bytes[ptr:ptr + n_bytes_arm_config])
    arm_configs["n_contexts"] = n_contexts
    ptr += n_bytes_arm_config

    # Read network configuration
    n_bytes_network_config = int.from_bytes(header_bytes[ptr:ptr + 1], byteorder='big', signed=False)
    ptr += 1
    network_configs = read_network_config(header_bytes[ptr:ptr + n_bytes_network_config])
    ptr += n_bytes_network_config

    # Read encoding configuration
    n_bytes_encoding_config = int.from_bytes(header_bytes[ptr:ptr + 1], byteorder='big', signed=False)
    ptr += 1
    encoding_configs = read_encoding_config(header_bytes[ptr:ptr + n_bytes_encoding_config])
    ptr += n_bytes_encoding_config

    acmax_scale_and_n_bytes = []
    for i in range(3):
        acmax_i = int.from_bytes(header_bytes[ptr:ptr + 2], byteorder='big', signed=False)
        ptr += 2
        if i < 2: # net and arm
            scale_i = struct.unpack_from("f", header_bytes, ptr)[0]
            ptr += 4
            n_bytes_i = int.from_bytes(header_bytes[ptr:ptr + 2], byteorder='big', signed=False)
            ptr += 2
        else: # encoding
            scale_i = 0
            n_bytes_i = int.from_bytes(header_bytes[ptr:ptr + 4], byteorder='big', signed=False)
            ptr += 4
        acmax_scale_and_n_bytes.append([acmax_i, scale_i, n_bytes_i])

    header: FrameHeader = {
        "n_bytes_header": n_bytes_header,
        "img_size": img_size,
        "n_contexts": n_contexts,
        "encoding_configs": encoding_configs,
        "network_configs": network_configs,
        "arm_configs": arm_configs,
        "acmax_scale_and_n_bytes": acmax_scale_and_n_bytes
    }

    print('\nContent of the frame header:')
    print('------------------------------')
    for k, v in header.items():
        print(f'{k:>20}: {v}')
    print('         ------------------------')

    return header
