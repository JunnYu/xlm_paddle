import paddle
import paddle.nn.functional as F
import numpy as np
import argparse
from reprod_log import ReprodLogger

from xlm_paddle import XLMForSequenceClassification, XLMTokenizer


def get_args(add_help=True):
    parser = argparse.ArgumentParser(
        description='PaddleNLP Classification Predict', add_help=add_help)

    parser.add_argument(
        "--text",
        default="You don't have to stay there.<sep>You can leave.",
        type=str,
        help="XNLI Premise <sep> Hypothesis Text")
    parser.add_argument(
        "--language", default="en", type=str, help="XNLI language")
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=True,
        help="Path of the trained model to be exported.", )
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        choices=["gpu", "cpu"],
        help="device", )
    args = parser.parse_args()
    return args


@paddle.no_grad()
def main(args):
    paddle.set_device(args.device)
    # define model
    model = XLMForSequenceClassification.from_pretrained(
        args.model_path, num_classes=3, dropout=0.0)
    tokenizer = XLMTokenizer.from_pretrained(args.model_path)
    tokenizer.current_lang = args.language
    model.eval()
    text_a, text_b = args.text.split("<sep>")
    tokenized_data = tokenizer(
        [text_a], [text_b],
        max_seq_len=256,
        return_token_type_ids=False,
        return_attention_mask=False)
    inputs = {
        "input_ids": paddle.to_tensor(
            tokenized_data["input_ids"], dtype="int64"),
    }
    inputs["langs"] = paddle.ones_like(
        inputs["input_ids"], dtype="int64") * tokenizer.lang2id[args.language]
    logits = model(**inputs)
    probs = F.softmax(logits, axis=-1).numpy()[0]
    label_id = probs.argmax()
    prob = probs[label_id]
    print(f"label_id: {label_id}, prob: {prob}")
    return label_id, prob


if __name__ == "__main__":
    args = get_args()
    label_id, prob = main(args)
    reprod_logger = ReprodLogger()
    reprod_logger.add("label_id", np.array([label_id]))
    reprod_logger.add("prob", np.array([prob]))
    reprod_logger.save("output_predict_engine.npy")
