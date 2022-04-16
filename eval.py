from paddle.metric import Accuracy
from xlm_paddle import XLMForSequenceClassification
from args import parse_args
from argparse import Namespace
from xlm.xnli_utils import XNLI_LANGS, XNLIDataset
from train import evaluate


def main(args):
    model = XLMForSequenceClassification.from_pretrained(
        args.pretrained_model_name_or_path, num_classes=3, dropout=0.0)
    metric = Accuracy()
    model.eval()
    params = Namespace(
        data_path=args.data_path,
        max_vocab=model.xlm.config["vocab_size"],
        min_count=0,
        tokens_per_batch=-1,
        max_batch_size=0,
        group_by_size=False,
        max_len=args.max_length,
        eos_index=model.xlm.config["eos_token_id"],
        pad_index=model.xlm.config["pad_token_id"],
        eval_batch_size=args.eval_batch_size,
        train_batch_size=args.train_batch_size,
        lang2id=model.xlm.config["lang2id"], )
    ds = XNLIDataset(params)
    results_dict = {}
    # val
    val_avg_acc = 0
    for lang in XNLI_LANGS:
        val_acc = evaluate(model, ds, lang, metric, val=True, params=params)
        results_dict[f"val_{lang}_acc"] = val_acc
        val_avg_acc += val_acc
        print(f"##########  val_{lang}_acc {val_acc} ##########")
    results_dict["val_avg_acc"] = val_avg_acc / 15

    # test
    test_avg_acc = 0
    for lang in XNLI_LANGS:
        test_acc = evaluate(model, ds, lang, metric, val=False, params=params)
        results_dict[f"test_{lang}_acc"] = test_acc
        test_avg_acc += test_acc
        print(f"##########  test_{lang}_acc {test_acc} ##########")
    results_dict["test_avg_acc"] = test_avg_acc / 15

    for k, v in results_dict.items():
        print(f"  {k} = {v}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
