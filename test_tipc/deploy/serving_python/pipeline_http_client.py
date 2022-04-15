# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import requests
import json


def get_args(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(
        description='Paddle Serving', add_help=add_help)

    parser.add_argument(
        "--text",
        default="You don't have to stay there.<sep>You can leave.",
        type=str,
        help="XNLI Premise <sep> Hypothesis Text")
    parser.add_argument(
        "--language", default="en", type=str, help="XNLI language")
    args = parser.parse_args()
    return args


def main(args):
    url = "http://127.0.0.1:18080/xlm/prediction"
    logid = 10000

    data = {
        "key": ["text", "language"],
        "value": [args.text, args.language],
        "logid": logid
    }
    # send requests
    r = requests.post(url=url, data=json.dumps(data))
    print(r.json())


if __name__ == "__main__":
    args = get_args()
    main(args)
