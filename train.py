import logging
import math
import os

import paddle
from argparse import Namespace
from paddle.amp import GradScaler, auto_cast
from paddle.metric import Accuracy
from paddle.optimizer import Adam, AdamW
from tqdm.auto import tqdm

from args import parse_args
from xlm_paddle import XLMForSequenceClassification
from xlm.utils import get_writer, save_json, set_seed, try_remove_old_ckpt
from xlm.xnli_utils import XNLI_LANGS, XNLIDataset, concat_batches, truncate

logger = logging.getLogger(__name__)


def totensor(t):
    return paddle.to_tensor(t)


@paddle.no_grad()
def evaluate(model, ds, lang, metric, val=True, params=""):
    model.eval()
    metric.reset()
    splt = "valid" if val else "test"
    lang_id = params.lang2id[lang]
    for batch in tqdm(ds.get_iterator(splt, lang), leave=False):
        (sent1, len1), (sent2, len2), idx = batch
        input_ids, lengths, position_ids, langs = concat_batches(
            sent1,
            len1,
            lang_id,
            sent2,
            len2,
            lang_id,
            params.pad_index,
            params.eos_index,
            reset_positions=False, )
        logits = model(
            input_ids=input_ids,
            lengths=lengths,
            langs=langs,
            position_ids=position_ids)
        correct = metric.compute(logits,
                                 totensor(ds.data[lang][splt]["y"][idx]))
        metric.update(correct)
    acc = metric.accumulate()
    model.train()
    return acc


def main(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                os.path.join(args.output_dir, "run.log"),
                mode="w",
                encoding="utf-8", )
        ], )
    logger.info("**********  Configuration Arguments **********")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"{arg}: {value}")
    logger.info("**************************************************")

    set_seed(args)

    writer = get_writer(args)

    model = XLMForSequenceClassification.from_pretrained(
        args.pretrained_model_name_or_path, num_classes=3, dropout=0.0)
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

    opt_cls = AdamW if args.optimizer.lower() == "adamw" else Adam
    optimizer = opt_cls(
        learning_rate=args.learning_rate,
        beta1=0.9,
        beta2=0.999,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(), )

    metric = Accuracy()

    if args.fp16:
        scaler = GradScaler(init_loss_scaling=args.scale_loss)

    num_update_steps_per_epoch = args.sentences_per_epoch // args.train_batch_size
    if args.max_train_steps > 0:
        args.num_train_epochs = math.ceil(args.max_train_steps /
                                          num_update_steps_per_epoch)
    else:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps

    logger.info("********** Running training **********")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous train batch size = {args.train_batch_size}")
    logger.info(f"  Instantaneous eval batch size = {args.eval_batch_size}")
    logger.info(
        f"  Total train batch size (w. accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    save_json(vars(args), os.path.join(args.output_dir, "args.json"))
    progress_bar = tqdm(range(args.max_train_steps))

    global_steps = 0
    tr_loss, logging_loss = 0.0, 0.0
    max_val_acc = 0.0

    dls = ds.get_iterator("train", "en")
    for epoch in range(args.num_train_epochs):
        step = 0
        for batch in dls:
            model.train()
            with auto_cast(
                    args.fp16,
                    custom_white_list=["layer_norm", "softmax", "gelu"]):
                (sent1, len1), (sent2, len2), idx = batch
                sent1, len1 = truncate(sent1, len1, params.max_len,
                                       params.eos_index)
                sent2, len2 = truncate(sent2, len2, params.max_len,
                                       params.eos_index)
                input_ids, lengths, position_ids, langs = concat_batches(
                    sent1,
                    len1,
                    params.lang2id["en"],
                    sent2,
                    len2,
                    params.lang2id["en"],
                    params.pad_index,
                    params.eos_index,
                    reset_positions=False, )
                outputs = model(
                    input_ids=input_ids,
                    lengths=lengths,
                    langs=langs,
                    position_ids=position_ids,
                    labels=totensor(ds.data["en"]["train"]["y"][idx]), )
                loss = outputs[0] / args.gradient_accumulation_steps
                tr_loss += loss.item()

            if args.fp16:
                scaled = scaler.scale(loss)
                scaled.backward()
            else:
                loss.backward()

            if step % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    scaler.minimize(optimizer, scaled)
                else:
                    optimizer.step()
                optimizer.clear_grad()
                progress_bar.update(1)
                global_steps += 1

                if (args.logging_steps > 0 and
                        global_steps % args.logging_steps == 0
                    ) or global_steps == args.max_train_steps:
                    writer.add_scalar(
                        "loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_steps, )
                    logger.info("global_steps {} loss: {:.8f}".format(
                        global_steps,
                        (tr_loss - logging_loss) / args.logging_steps, ))
                    logging_loss = tr_loss

                if (args.save_steps > 0 and global_steps % args.save_steps == 0
                    ) or global_steps == args.max_train_steps:
                    logger.info("********** Running evaluating **********")
                    results_dict = {}
                    # val
                    val_avg_acc = 0
                    for lang in XNLI_LANGS:
                        val_acc = evaluate(
                            model, ds, lang, metric, val=True, params=params)
                        results_dict[f"val_{lang}_acc"] = val_acc
                        val_avg_acc += val_acc
                        logger.info(
                            f"##########  val_{lang}_acc {val_acc} ##########")
                        print(
                            f"##########  val_{lang}_acc {val_acc} ##########")

                    results_dict["val_avg_acc"] = val_avg_acc / 15

                    # test
                    test_avg_acc = 0
                    for lang in XNLI_LANGS:
                        test_acc = evaluate(
                            model, ds, lang, metric, val=False, params=params)
                        results_dict[f"test_{lang}_acc"] = test_acc
                        test_avg_acc += test_acc
                        logger.info(
                            f"##########  test_{lang}_acc {test_acc} ##########"
                        )
                        print(
                            f"##########  test_{lang}_acc {test_acc} ##########"
                        )
                    results_dict["test_avg_acc"] = test_avg_acc / 15

                    for k, v in results_dict.items():
                        writer.add_scalar(f"eval/{k}", v, global_steps)
                        logger.info(f"  {k} = {v}")
                        print(f"  {k} = {v}")

                    val_avg_acc = results_dict["val_avg_acc"]
                    test_avg_acc = results_dict["test_avg_acc"]

                    if val_avg_acc >= max_val_acc:
                        max_val_acc = val_avg_acc
                        logger.info(
                            f"########## Step {global_steps} val_avg_acc {max_val_acc} test_avg_acc {test_avg_acc} ##########"
                        )
                        print(
                            f"########## Step {global_steps} val_avg_acc {max_val_acc} test_avg_acc {test_avg_acc} ##########"
                        )

                    output_dir = os.path.join(
                        args.output_dir,
                        "ckpt",
                        f"step-{global_steps}-test_avg_acc-{test_avg_acc}-val_avg_acc-{val_avg_acc}",
                    )
                    os.makedirs(output_dir, exist_ok=True)
                    model.save_pretrained(output_dir)
                    try_remove_old_ckpt(args.output_dir, topk=args.topk)

                    logger.info("********** Evaluating Done **********")

            if global_steps >= args.max_train_steps:
                return

            step += 1
            if step >= args.sentences_per_epoch // args.train_batch_size:
                print(f"Epoch {epoch} resample!")
                dls = ds.get_iterator("train", "en")
                break


if __name__ == "__main__":
    args = parse_args()
    main(args)
