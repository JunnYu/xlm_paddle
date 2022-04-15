# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict
import argparse

donot_transpose = [
    ".layer_norm", ".position_embeddings.", ".lang_embeddings.", ".embeddings."
]


def convert_pytorch_checkpoint_to_paddle(pytorch_checkpoint_path,
                                         paddle_dump_path):

    import torch
    import paddle
    pytorch_state_dict = torch.load(
        pytorch_checkpoint_path, map_location="cpu")
    paddle_state_dict = OrderedDict()
    for k, v in pytorch_state_dict.items():
        is_transpose = False
        if k[-7:] == ".weight":
            if not any(d in k for d in donot_transpose):
                if v.ndim == 2:
                    v = v.transpose(0, 1)
                    is_transpose = True
        oldk = k
        k = k.replace("transformer", "xlm")
        # remove pred_layer.proj.weight
        if "pred_layer.proj.weight" in k:
            continue
        if "pred_layer.proj.bias" in k:
            k = k.replace(".proj.", ".")
        print(f"Converting: {oldk} => {k} is_transpose {is_transpose}")
        paddle_state_dict[k] = v.data.numpy().astype("float32")

    paddle.save(paddle_state_dict, paddle_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # from os import system, chdir
    # chdir("xlm-mlm-tlm-xnli15-1024")
    # ok = system("wget https://huggingface.co/xlm-mlm-tlm-xnli15-1024/resolve/main/pytorch_model.bin")
    # chdir("../")
    parser.add_argument(
        "--pytorch_checkpoint_path",
        default="xlm-mlm-tlm-xnli15-1024/pytorch_model.bin",
        type=str,
        required=False,
        help="Path to the Pytorch checkpoint path.")
    parser.add_argument(
        "--paddle_dump_path",
        default="xlm-mlm-tlm-xnli15-1024/model_state.pdparams",
        type=str,
        required=False,
        help="Path to the output Paddle model.")
    args = parser.parse_args()
    convert_pytorch_checkpoint_to_paddle(args.pytorch_checkpoint_path,
                                         args.paddle_dump_path)
