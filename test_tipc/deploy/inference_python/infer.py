import os
from paddle import inference
import numpy as np
from scipy.special import softmax
from xlm_paddle import XLMTokenizer
from paddlenlp.data import Pad


class InferenceEngine(object):
    """InferenceEngine

    Inference engina class which contains preprocess, run, postprocess

    """

    def __init__(self, args):
        """
        Args:
            args: Parameters generated using argparser.

        Returns: None
        """
        super().__init__()
        self.args = args

        # init inference engine
        self.predictor, self.config, self.input_tensors, self.output_tensors = self.load_predictor(
            os.path.join(args.model_dir, "inference.pdmodel"),
            os.path.join(args.model_dir, "inference.pdiparams"))

        # build transforms
        self.tokenizer = XLMTokenizer.from_pretrained(args.model_dir)
        self.batchify_fn = Pad(axis=0,
                               pad_val=self.tokenizer.pad_token_id,
                               dtype="int64")

        # wamrup
        if self.args.warmup > 0:
            for idx in range(args.warmup):
                input_ids = np.random.randint(
                    5, 200, size=(4, 32)).astype("int64")
                langs = np.ones_like(input_ids).astype(
                    "int64") * self.tokenizer.lang2id["en"]
                self.input_tensors[0].copy_from_cpu(input_ids)
                self.input_tensors[1].copy_from_cpu(langs)
                self.predictor.run()
                self.output_tensors[0].copy_to_cpu()
        return

    def load_predictor(self, model_file_path, params_file_path):
        """load_predictor

        initialize the inference engine

        Args:
            model_file_path: inference model path (*.pdmodel)
            model_file_path: inference parmaeter path (*.pdiparams)
        Return:
            predictor: Predictor created using Paddle Inference.
            config: Configuration of the predictor.
            input_tensor: Input tensor of the predictor.
            output_tensor: Output tensor of the predictor.
        """
        args = self.args
        config = inference.Config(model_file_path, params_file_path)
        if args.use_gpu:
            pass
            # config.enable_use_gpu(9830, 0)
        else:
            config.disable_gpu()

        # enable memory optim
        config.enable_memory_optim()
        config.disable_glog_info()

        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(True)

        # create predictor
        predictor = inference.create_predictor(config)

        # get input and output tensor property
        input_tensors = [
            predictor.get_input_handle(name)
            for name in predictor.get_input_names()
        ]
        output_tensors = [
            predictor.get_output_handle(name)
            for name in predictor.get_output_names()
        ]

        return predictor, config, input_tensors, output_tensors

    def preprocess(self, text_a, text_b, language="en"):
        """preprocess

        Preprocess to the input.

        Args:
            text: Text.

        Returns: Input data after preprocess.
        """
        self.tokenizer.current_lang = language
        data = self.tokenizer(
            [text_a], [text_b],
            max_seq_len=256,
            return_attention_mask=False,
            return_token_type_ids=False)
        input_ids = self.batchify_fn(data["input_ids"])
        langs = np.ones_like(input_ids).astype("int64") * \
            self.tokenizer.lang2id[language]
        return input_ids, langs

    def postprocess(self, x):
        """postprocess

        Postprocess to the inference engine output.

        Args:
            x: Inference engine output.

        Returns: Output data after argmax.
        """
        score = softmax(x[0], axis=-1)
        label_id = score.argmax()
        prob = score[label_id]
        return label_id, prob

    def run(self, input_ids, langs):
        """run

        Inference process using inference engine.

        Args:
            x: Input data after preprocess.

        Returns: Inference engine output
        """
        self.input_tensors[0].copy_from_cpu(input_ids)
        self.input_tensors[1].copy_from_cpu(langs)
        self.predictor.run()
        output = self.output_tensors[0].copy_to_cpu()
        return output


def get_args(add_help=True):
    """
    parse args
    """
    import argparse

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser(
        description="PaddleNLP Classification Training", add_help=add_help)
    parser.add_argument(
        "--text",
        default="You don't have to stay there.<sep>You can leave.",
        type=str,
        help="XNLI Premise <sep> Hypothesis Text")
    parser.add_argument(
        "--language", default="en", type=str, help="XNLI language")
    parser.add_argument(
        "--model_dir", default=None, help="inference model dir")
    parser.add_argument(
        "--use_gpu", default=False, type=str2bool, help="use_gpu")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument(
        "--benchmark", default=False, type=str2bool, help="benchmark")
    parser.add_argument("--warmup", default=0, type=int, help="warmup iter")

    args = parser.parse_args()
    return args


def infer_main(args):
    """infer_main

    Main inference function.

    Args:
        args: Parameters generated using argparser.

    Returns:
        label_id: Class index of the input.
        prob: : Probability of the input.
    """
    inference_engine = InferenceEngine(args)

    # init benchmark
    if args.benchmark:
        import auto_log
        autolog = auto_log.AutoLogger(
            model_name="classification",
            batch_size=args.batch_size,
            inference_config=inference_engine.config,
            gpu_ids="auto" if args.use_gpu else None)

    assert args.batch_size == 1, "batch size just supports 1 now."

    # enable benchmark
    if args.benchmark:
        autolog.times.start()

    text_a, text_b = args.text.split("<sep>")
    # preprocess
    input_ids, langs = inference_engine.preprocess(
        text_a, text_b, language=args.language)

    if args.benchmark:
        autolog.times.stamp()

    output = inference_engine.run(input_ids, langs)

    if args.benchmark:
        autolog.times.stamp()

    # postprocess
    label_id, prob = inference_engine.postprocess(output)

    if args.benchmark:
        autolog.times.stamp()
        autolog.times.end(stamp=True)
        autolog.report()

    print(f"text: {args.text}, label_id: {label_id}, prob: {prob}")
    return label_id, prob


if __name__ == "__main__":
    args = get_args()
    label_id, prob = infer_main(args)
    from reprod_log import ReprodLogger
    print(label_id, prob)
    reprod_logger = ReprodLogger()
    reprod_logger.add("label_id", np.array([label_id]))
    reprod_logger.add("prob", np.array([prob]))
    reprod_logger.save("output_inference_engine.npy")
