# XLM

## 目录

- [0. 仓库结构]()
- [1. 简介]()
- [2. 数据集和复现精度]()
- [3. 准备数据与环境]()
    - [3.1 准备环境]()
    - [3.2 准备数据]()
    - [3.3 准备模型]()
- [4. 开始使用]()
    - [4.1 模型训练]()
    - [4.2 模型评估]()
    - [4.3 模型预测]()
- [5. 模型推理部署]()
    - [5.1 基于Inference的推理]()
    - [5.2 基于Serving的服务化部署]()
- [6. TIPC自动化测试脚本]()
- [7. 注意]()
- [8. LICENSE]()
- [9. 参考链接与文献]()

## 0. 仓库结构
```bash
root:[./]
|--.pre-commit-config.yaml
|--args.py
|--deploy
|      |--inference_python
|      |      |--infer.py
|      |      |--README.md
|      |      |--xlm_paddle
|      |      |      |--adaptive.py
|      |      |      |--modeling.py
|      |      |      |--tokenizer.py
|      |      |      |--__init__.py
|      |--serving_python
|      |      |--config.yml
|      |      |--PipelineServingLogs
|      |      |      |--pipeline.log
|      |      |      |--pipeline.log.wf
|      |      |      |--pipeline.tracer
|      |      |--pipeline_http_client.py
|      |      |--ProcessInfo.json
|      |      |--README.md
|      |      |--web_service.py
|      |      |--xlm_client
|      |      |      |--serving_client_conf.prototxt
|      |      |      |--serving_client_conf.stream.prototxt
|      |      |--xlm_paddle
|      |      |      |--adaptive.py
|      |      |      |--modeling.py
|      |      |      |--tokenizer.py
|      |      |      |--__init__.py
|      |      |--xlm_server
|      |      |      |--inference.pdiparams
|      |      |      |--inference.pdmodel
|      |      |      |--serving_server_conf.prototxt
|      |      |      |--serving_server_conf.stream.prototxt
|--images
|      |--py_serving_client_results.jpg
|      |--py_serving_startup_visualization.jpg
|--output_inference_engine.npy
|--print_project_tree.py
|--README.md
|--requirements.txt
|--shell
|      |--export.sh
|      |--inference_python.sh
|      |--predict.sh
|      |--train.sh
|--test_tipc
|      |--common_func.sh
|      |--configs
|      |      |--XLM
|      |      |      |--model_linux_gpu_normal_normal_serving_python_linux_gpu_cpu.txt
|      |      |      |--train_infer_python.txt
|      |--docs
|      |      |--test_serving.md
|      |      |--test_train_inference_python.md
|      |      |--tipc_guide.png
|      |      |--tipc_serving.jpg
|      |      |--tipc_train_inference.jpg
|      |--output
|      |      |--python_infer_cpu_usemkldnn_False_threads_null_precision_null_batchsize_null.log
|      |      |--python_infer_gpu_usetrt_null_precision_null_batchsize_null.log
|      |      |--results_python.log
|      |      |--results_serving.log
|      |      |--server_infer_gpu_pipeline_http_usetrt_null_precision_null_batchsize_1.log
|      |--README.md
|      |--test_serving.sh
|      |--test_train_inference_python.sh
|--tools
|      |--export_model.py
|      |--predict.py
|      |--xlm_paddle
|      |      |--adaptive.py
|      |      |--modeling.py
|      |      |--tokenizer.py
|      |      |--__init__.py
|--train.py
|--xlm
|      |--data
|      |      |--dictionary.py
|      |      |--XNLI
|      |      |      |--test.label.ar
|      |      |      |--test.label.bg
|      |      |      |--test.label.de
|      |      |      |--test.label.el
|      |      |      |--test.label.en
|      |      |      |--test.label.es
|      |      |      |--test.label.fr
|      |      |      |--test.label.hi
|      |      |      |--test.label.ru
|      |      |      |--test.label.sw
|      |      |      |--test.label.th
|      |      |      |--test.label.tr
|      |      |      |--test.label.ur
|      |      |      |--test.label.vi
|      |      |      |--test.label.zh
|      |      |      |--test.s1.ar
|      |      |      |--test.s1.ar.pkl
|      |      |      |--test.s1.bg
|      |      |      |--test.s1.bg.pkl
|      |      |      |--test.s1.de
|      |      |      |--test.s1.de.pkl
|      |      |      |--test.s1.el
|      |      |      |--test.s1.el.pkl
|      |      |      |--test.s1.en
|      |      |      |--test.s1.en.pkl
|      |      |      |--test.s1.es
|      |      |      |--test.s1.es.pkl
|      |      |      |--test.s1.fr
|      |      |      |--test.s1.fr.pkl
|      |      |      |--test.s1.hi
|      |      |      |--test.s1.hi.pkl
|      |      |      |--test.s1.ru
|      |      |      |--test.s1.ru.pkl
|      |      |      |--test.s1.sw
|      |      |      |--test.s1.sw.pkl
|      |      |      |--test.s1.th
|      |      |      |--test.s1.th.pkl
|      |      |      |--test.s1.tr
|      |      |      |--test.s1.tr.pkl
|      |      |      |--test.s1.ur
|      |      |      |--test.s1.ur.pkl
|      |      |      |--test.s1.vi
|      |      |      |--test.s1.vi.pkl
|      |      |      |--test.s1.zh
|      |      |      |--test.s1.zh.pkl
|      |      |      |--test.s2.ar
|      |      |      |--test.s2.ar.pkl
|      |      |      |--test.s2.bg
|      |      |      |--test.s2.bg.pkl
|      |      |      |--test.s2.de
|      |      |      |--test.s2.de.pkl
|      |      |      |--test.s2.el
|      |      |      |--test.s2.el.pkl
|      |      |      |--test.s2.en
|      |      |      |--test.s2.en.pkl
|      |      |      |--test.s2.es
|      |      |      |--test.s2.es.pkl
|      |      |      |--test.s2.fr
|      |      |      |--test.s2.fr.pkl
|      |      |      |--test.s2.hi
|      |      |      |--test.s2.hi.pkl
|      |      |      |--test.s2.ru
|      |      |      |--test.s2.ru.pkl
|      |      |      |--test.s2.sw
|      |      |      |--test.s2.sw.pkl
|      |      |      |--test.s2.th
|      |      |      |--test.s2.th.pkl
|      |      |      |--test.s2.tr
|      |      |      |--test.s2.tr.pkl
|      |      |      |--test.s2.ur
|      |      |      |--test.s2.ur.pkl
|      |      |      |--test.s2.vi
|      |      |      |--test.s2.vi.pkl
|      |      |      |--test.s2.zh
|      |      |      |--test.s2.zh.pkl
|      |      |      |--train.label.en
|      |      |      |--train.s1.en
|      |      |      |--train.s1.en.pkl
|      |      |      |--train.s2.en
|      |      |      |--train.s2.en.pkl
|      |      |      |--valid.label.ar
|      |      |      |--valid.label.bg
|      |      |      |--valid.label.de
|      |      |      |--valid.label.el
|      |      |      |--valid.label.en
|      |      |      |--valid.label.es
|      |      |      |--valid.label.fr
|      |      |      |--valid.label.hi
|      |      |      |--valid.label.ru
|      |      |      |--valid.label.sw
|      |      |      |--valid.label.th
|      |      |      |--valid.label.tr
|      |      |      |--valid.label.ur
|      |      |      |--valid.label.vi
|      |      |      |--valid.label.zh
|      |      |      |--valid.s1.ar
|      |      |      |--valid.s1.ar.pkl
|      |      |      |--valid.s1.bg
|      |      |      |--valid.s1.bg.pkl
|      |      |      |--valid.s1.de
|      |      |      |--valid.s1.de.pkl
|      |      |      |--valid.s1.el
|      |      |      |--valid.s1.el.pkl
|      |      |      |--valid.s1.en
|      |      |      |--valid.s1.en.pkl
|      |      |      |--valid.s1.es
|      |      |      |--valid.s1.es.pkl
|      |      |      |--valid.s1.fr
|      |      |      |--valid.s1.fr.pkl
|      |      |      |--valid.s1.hi
|      |      |      |--valid.s1.hi.pkl
|      |      |      |--valid.s1.ru
|      |      |      |--valid.s1.ru.pkl
|      |      |      |--valid.s1.sw
|      |      |      |--valid.s1.sw.pkl
|      |      |      |--valid.s1.th
|      |      |      |--valid.s1.th.pkl
|      |      |      |--valid.s1.tr
|      |      |      |--valid.s1.tr.pkl
|      |      |      |--valid.s1.ur
|      |      |      |--valid.s1.ur.pkl
|      |      |      |--valid.s1.vi
|      |      |      |--valid.s1.vi.pkl
|      |      |      |--valid.s1.zh
|      |      |      |--valid.s1.zh.pkl
|      |      |      |--valid.s2.ar
|      |      |      |--valid.s2.ar.pkl
|      |      |      |--valid.s2.bg
|      |      |      |--valid.s2.bg.pkl
|      |      |      |--valid.s2.de
|      |      |      |--valid.s2.de.pkl
|      |      |      |--valid.s2.el
|      |      |      |--valid.s2.el.pkl
|      |      |      |--valid.s2.en
|      |      |      |--valid.s2.en.pkl
|      |      |      |--valid.s2.es
|      |      |      |--valid.s2.es.pkl
|      |      |      |--valid.s2.fr
|      |      |      |--valid.s2.fr.pkl
|      |      |      |--valid.s2.hi
|      |      |      |--valid.s2.hi.pkl
|      |      |      |--valid.s2.ru
|      |      |      |--valid.s2.ru.pkl
|      |      |      |--valid.s2.sw
|      |      |      |--valid.s2.sw.pkl
|      |      |      |--valid.s2.th
|      |      |      |--valid.s2.th.pkl
|      |      |      |--valid.s2.tr
|      |      |      |--valid.s2.tr.pkl
|      |      |      |--valid.s2.ur
|      |      |      |--valid.s2.ur.pkl
|      |      |      |--valid.s2.vi
|      |      |      |--valid.s2.vi.pkl
|      |      |      |--valid.s2.zh
|      |      |      |--valid.s2.zh.pkl
|      |--utils.py
|      |--xnli_utils.py
|      |--__init__.py
|--xlm_paddle
|      |--adaptive.py
|      |--modeling.py
|      |--tokenizer.py
|      |--__init__.py
```

## 1. 简介
**论文:** [Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291)

注意力模型，尤其是BERT模型，已经在NLP领域分类和翻译任务上取得了极具前景的结果。Facebook AI在一篇新论文中提出了一个改良版的BERT模型XLM，在以上两个任务上均取得了SOTA结果。
XLM用了一个常见的预处理技术BPE（byte pair encoder字节对编码）以及BERT双语言训练机制来学习不同语言中词与词之间的关系。这个模型在跨语言分类任务（15个语言的句子蕴含任务）上比其他模型取得了更好的效果，并且显著提升了有预训练的机器翻译效果。

<div align="center">
    <img src="./images/xlm_framework.jpg" width=800">
</div>

## 2. 数据集和复现精度

数据集为跨语言的`XNLI`。

| 模型      | test avg acc (论文精度)  | test avg acc (复现精度) |
|:---------:|:----------:|:----------:|
| XLM | 0.751   | 0.747 |

复现结果想要达到论文中的结果是很难的，因此通过实际训练得到最终的效果不一定会很稳定，本项目通过大量实验，才最终调到这个最优的结果。
- （1）由于XLM模型的训练过程中进行了随机采样，即在数据集上仅采样了20000条句子，采样的方法会导致最终的结果会有所波动。（采样好的话，效果不错，采样差的话效果就不太行。）
- （2）XLM模型对超参数非常敏感，许多人使用官方代码都无法复现出论文的结果[这里是相关的issue](https://github.com/facebookresearch/XLM/search?q=xnli+reproduce&type=issues) [issue1说平均精度低了2个百分点](https://github.com/facebookresearch/XLM/issues/281) [issue2中平均精度只有71.2](https://github.com/facebookresearch/XLM/issues/199)
- 作者在issuse中指出他们汇报的结果是进行了一次实验而得到的，并未进行多次取平均。

## 3. 准备环境与数据

### 3.1 准备环境

* 下载代码

```bash
git clone https://github.com/junnyu/xlm_paddle.git
cd test_tipc
```

* 安装paddlepaddle

```bash
# 需要安装2.2及以上版本的Paddle，如果
# 安装GPU版本的Paddle
pip install paddlepaddle-gpu==2.2.0
# 安装CPU版本的Paddle
pip install paddlepaddle==2.2.0
```

更多安装方法可以参考：[Paddle安装指南](https://www.paddlepaddle.org.cn/)。

* 安装requirements

```bash
pip install -r requirements.txt
# 如果想要对特殊语言进行tokenizer，例如：泰语、日语，那么请根据错误提示，安装第三方库，`xlm_paddle/tokenizer.py`文件中有相关的信息。
```

### 3.2 准备数据

- 为了确保使用的数据集与原论文一致，因此代码1参照`facebook/xlm`仓库中的数据处理过程。
- 本人将处理好的数据转化成了`pkl`文件，可查看文件夹`xlm/data/XNLI`。
- 处理过程如下
```bash
# 这些代码可以在`Google Colab`中运行
!unzip XLM-main.zip
!chmod -R 777 XLM-main/
%cd XLM-main
# Thai
!pip install pythainlp
# Chinese
%cd tools/
!wget https://nlp.stanford.edu/software/stanford-segmenter-2018-10-16.zip
!unzip stanford-segmenter-2018-10-16.zip
%cd ../
# 下载处理数据
!./get-data-xnli.sh
!./prepare-xnli.sh
# 压缩下载
!mv ./data/processed/XLM15/eval/XNLI ./XNLI
!tar -zcvf xnli.tar.gz ./XNLI
# 然后下载将这个放进`xlm/data/XNLI`文件夹。
```

### 3.3 准备模型

如果您希望直接体验评估或者预测推理过程，可以直接根据第2章的内容下载提供的预训练模型，直接体验模型评估、预测、推理部署等内容。


## 4. 开始使用

### 4.1 模型训练

* 单机单卡训练

```bash
python train.py --pretrained_model_name_or_path xlm-mlm-tlm-xnli15-1024 --output_dir ./xnli_outputs
```

部分训练日志如下所示。

```
04/12/2022 22:47:08 - INFO - __main__ -   val_ar_acc = 0.7172690763052209
04/12/2022 22:47:08 - INFO - __main__ -   val_bg_acc = 0.7606425702811245
04/12/2022 22:47:08 - INFO - __main__ -   val_de_acc = 0.7767068273092369
04/12/2022 22:47:08 - INFO - __main__ -   val_el_acc = 0.7522088353413655
04/12/2022 22:47:08 - INFO - __main__ -   val_en_acc = 0.8437751004016064
04/12/2022 22:47:08 - INFO - __main__ -   val_es_acc = 0.7891566265060241
04/12/2022 22:47:08 - INFO - __main__ -   val_fr_acc = 0.7811244979919679
04/12/2022 22:47:08 - INFO - __main__ -   val_hi_acc = 0.6899598393574298
04/12/2022 22:47:08 - INFO - __main__ -   val_ru_acc = 0.7429718875502008
04/12/2022 22:47:08 - INFO - __main__ -   val_sw_acc = 0.6714859437751004
04/12/2022 22:47:08 - INFO - __main__ -   val_th_acc = 0.706425702811245
04/12/2022 22:47:08 - INFO - __main__ -   val_tr_acc = 0.7208835341365462
04/12/2022 22:47:08 - INFO - __main__ -   val_ur_acc = 0.6461847389558233
04/12/2022 22:47:08 - INFO - __main__ -   val_vi_acc = 0.7397590361445783
04/12/2022 22:47:08 - INFO - __main__ -   val_zh_acc = 0.751004016064257
04/12/2022 22:47:08 - INFO - __main__ -   val_avg_acc = 0.7393038821954484
04/12/2022 22:47:08 - INFO - __main__ -   test_ar_acc = 0.7381237524950099
04/12/2022 22:47:08 - INFO - __main__ -   test_bg_acc = 0.7758483033932135
04/12/2022 22:47:08 - INFO - __main__ -   test_de_acc = 0.7686626746506986
04/12/2022 22:47:08 - INFO - __main__ -   test_el_acc = 0.7676646706586826
04/12/2022 22:47:08 - INFO - __main__ -   test_en_acc = 0.846307385229541
04/12/2022 22:47:08 - INFO - __main__ -   test_es_acc = 0.7978043912175649
04/12/2022 22:47:08 - INFO - __main__ -   test_fr_acc = 0.7924151696606786
04/12/2022 22:47:08 - INFO - __main__ -   test_hi_acc = 0.6880239520958084
04/12/2022 22:47:08 - INFO - __main__ -   test_ru_acc = 0.7618762475049901
04/12/2022 22:47:08 - INFO - __main__ -   test_sw_acc = 0.691816367265469
04/12/2022 22:47:08 - INFO - __main__ -   test_th_acc = 0.7109780439121757
04/12/2022 22:47:08 - INFO - __main__ -   test_tr_acc = 0.7173652694610778
04/12/2022 22:47:08 - INFO - __main__ -   test_ur_acc = 0.6584830339321357
04/12/2022 22:47:08 - INFO - __main__ -   test_vi_acc = 0.7447105788423154
04/12/2022 22:47:08 - INFO - __main__ -   test_zh_acc = 0.7481037924151697
04/12/2022 22:47:08 - INFO - __main__ -   test_avg_acc = 0.7472122421823022
```

* 单机多卡训练
TODO

更多配置参数可以参考[train.py](./train.py)的`parse_args`函数。

### 4.2 模型评估

该项目中，训练与评估脚本同时进行，请查看训练过程中的评价指标。

### 4.3 模型预测

* 使用GPU预测

```
python tools/predict.py --model_path=./xnli_outputs/ckpt
```

对于下面的文本进行预测

`You don't have to stay there.<sep>You can leave.` 其中 <sep>是为了便于切分文本

最终输出结果为`label_id: 0, prob: 0.5407765507698059`，表示预测的标签ID是`0`，置信度为`0.5407765507698059`。

* 使用CPU预测

```
python tools/predict.py --model_path=./xnli_outputs/ckpt --device=cpu
```
对于下面的文本进行预测

`You don't have to stay there.<sep>You can leave.` 其中 <sep>是为了便于切分文本

最终输出结果为`label_id: 0, prob: 0.5407765507698059`，表示预测的标签ID是`0`，置信度为`0.5407765507698059`。

## 5. 模型推理部署

### 5.1 基于Inference的推理

Inference推理教程可参考：[链接](./deploy/inference_python/README.md)。

### 5.2 基于Serving的服务化部署

Serving部署教程可参考：[链接](deploy/serving_python/README.md)。


## 6. TIPC自动化测试脚本

以Linux基础训练推理测试为例，测试流程如下。

* 运行测试命令

```bash
bash test_tipc/test_train_inference_python.sh test_tipc/configs/XLM/train_infer_python.txt lite_train_whole_infer
```

如果运行成功，在终端中会显示下面的内容，具体的日志也会输出到`test_tipc/output/`文件夹中的文件中。

```
[33m Run successfully with command - python train.py --save_steps 2500      --max_steps=2500           !  [0m
[33m Run successfully with command - python tools/export_model.py --model_path=./xnli_outputs/ckpt --save_inference_dir ./xlm_infer      !  [0m
[33m Run successfully with command - python deploy/inference_python/infer.py --model_dir ./xlm_infer --use_gpu=True               > ./test_tipc/output/python_infer_gpu_usetrt_null_precision_null_batchsize_null.log 2>&1 !  [0m
[33m Run successfully with command - python deploy/inference_python/infer.py --model_dir ./xlm_infer --use_gpu=False --benchmark=False               > ./test_tipc/output/python_infer_cpu_usemkldnn_False_threads_null_precision_null_batchsize_null.log 2>&1 !  [0m
```



* 更多详细内容，请参考：[XLM TIPC测试文档](./test_tipc/README.md)。
* 如果运行失败，可以先根据报错的具体命令，自查下配置文件是否正确，如果无法解决，可以给Paddle提ISSUE：[https://github.com/PaddlePaddle/Paddle/issues/new/choose](https://github.com/PaddlePaddle/Paddle/issues/new/choose)；如果您在微信群里的话，也可以在群里及时提问。


## 8. LICENSE

本项目的发布受[Apache 2.0 license](./LICENSE)许可认证。

## 9. 参考链接与文献

TODO
