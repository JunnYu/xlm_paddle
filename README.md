# Cross-lingual Language Model Pretraining
- paddle2.x verison XLM
- paddle2.x 版本的XLM模型

## 目录

- [1. 简介](#1-简介)
- [2. 数据集和复现精度](#2-数据集和复现精度)
- [3. 准备数据与环境](#3-数据集和复现精度)
    - [3.1 准备环境](#31-准备环境)
    - [3.2 准备数据](#32-准备数据)
    - [3.3 准备模型](#33-准备模型)
- [4. 开始使用](#4-开始使用)
    - [4.1 模型训练](#41-模型训练)
    - [4.2 模型评估](#42-模型评估)
    - [4.3 模型预测](#43-模型预测)
- [5. 模型推理部署](#5-模型推理部署)
    - [5.1 基于Inference的推理](#51-基于Inference的推理)
    - [5.2 基于Serving的服务化部署](#52-基于Serving的服务化部署)
- [6. 自动化测试脚本](#6-自动化测试脚本)
- [7. LICENSE](#7-LICENSE)
- [8. 参考链接与文献](#8-参考链接与文献)


## 1. 简介

**摘要：**
最近的研究证明了生成式预训练对英语自然语言理解的有效性。在这项工作中，我们将这种方法扩展到多种语言，并展示了跨语言预训练的有效性。我们提出了两种学习跨语言语言模型 (XLM) 的方法：一种是仅依赖单语数据的无监督方法，另一种是利用具有新的跨语言语言模型目标的并行数据的监督方法。我们在跨语言分类、无监督和有监督机器翻译方面获得了最先进的结果。在 XNLI 上，我们的方法以 4.9% 的绝对精度提升了最新技术水平。在无监督机器翻译上，我们在 WMT'16 German-English 上获得 34.3 BLEU，将之前的最新技术提高了 9 BLEU 以上。在有监督的机器翻译上，我们在 WMT'16 罗马尼亚语-英语上获得了 38.5 BLEU 的最新技术水平，比之前的最佳方法高出 4 BLEU 以上。

**论文:** [Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291)

**参考repo:** [facebookresearch/XLM](https://github.com/facebookresearch/XLM) 和 [huggingface/transformers](https://github.com/huggingface/transformers/tree/main/src/transformers/models/xlm)

在此非常感谢`facebookresearch/XLM`的issue，提高了本repo复现论文的效率。

**aistudio体验教程:** TODO

## 2. 数据集和复现精度

数据集云处理：
数据集处理过程参考了[facebookresearch/XLM](https://github.com/facebookresearch/XLM)仓库，以下代码可以在`Google Colab`中运行。 将这个`GOOGLE-COLAB-PROCESS.ipynb`上传到colab中，并运行。

```bash
# 将XLM-main.zip和GOOGLE-COLAB-PROCESS.ipynb上传到`Google Colab`。
unzip XLM-main.zip
chmod -R 777 XLM-main/
cd XLM-main
# Thai
pip install pythainlp
# Chinese
cd tools/
wget https://nlp.stanford.edu/software/stanford-segmenter-2018-10-16.zip
unzip stanford-segmenter-2018-10-16.zip
cd ../
# 下载处理数据
./get-data-xnli.sh
./prepare-xnli.sh
# 压缩下载
mv ./data/processed/XLM15/eval/XNLI ./XNLI
tar -zcvf xnli.tar.gz ./XNLI
# 我们将处理好的数据./XNLI放进xlm/data/XNLI/这里存放转换后的数据集.txt
```

格式如下：

- 数据集大小：XNLI将NLI数据集扩展到15种语言，包括英语、法语、西班牙语、德语、希腊语、保加利亚语、俄语、土耳其语、阿拉伯语、越南语、泰语、中文、印地语、斯瓦希里语和乌尔都语，并以NLI的三分类格式为每种语言分别提供了7500个经人工标注的开发和测试实例，合计112500个标准句子对。
- 数据集下载链接：[aistudio上处理后的XNLI数据集](https://aistudio.baidu.com/aistudio/datasetdetail/139402)
- 数据格式：原始数据是`tsv格式`，这里提供的是处理好的`pkl格式`，专门为了复现论文而提供的。


| 标题       | 信息                                                        |
| ---------------- | ------------------------------------------------------------ |
| 论文中精度       | 75.1                                                         |
| 参考代码的精度   | 73.5 # 实际效果达不到论文中的指标                                     |
| 本repo复现的精度 | 74.7 # 实际效果达不到论文中的指标                                       |
| 数据集名称       | XNLI                                                         |
| 模型下载链接     | [原始权重](https://huggingface.co/junnyu/xlm-mlm-tlm-xnli15-1024-paddle ) 和 [微调后的权重](https://huggingface.co/junnyu/xlm-mlm-tlm-xnli15-1024-paddle-fintuned-on-xnli) |




## 3. 准备数据与环境


### 3.1 准备环境

- **硬件：** RTX3090 24G
- **框架：**
  - PaddlePaddle >= 2.2.0

**依赖：**

- python -m pip install paddlepaddle-gpu==2.2.2.post112 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
- paddlenlp
- reprod-log

直接使用`pip install -r requirements.txt`安装依赖即可。

### 3.2 准备数据

参考 **[2. 数据集和复现精度]**

将处理好的数据`xnli.tar.gz`下载至本地，然后放进`xlm/data/XNLI`文件夹。

### 3.3 准备模型

**（1）模型权重pytorch->paddle转换**

```bash
cd xlm-mlm-tlm-xnli15-1024
wget https://huggingface.co/xlm-mlm-tlm-xnli15-1024/resolve/main/pytorch_model.bin
cd ../
python convert.py
```
**（2）或者直接下载huggingface的模型**

- [原始权重](https://huggingface.co/junnyu/xlm-mlm-tlm-xnli15-1024-paddle )
- [微调后的权重](https://huggingface.co/junnyu/xlm-mlm-tlm-xnli15-1024-paddle-fintuned-on-xnli)

```python
from xlm_paddle import XLMForSequenceClassification, XLMTokenizer
model = XLMForSequenceClassification.from_pretrained("xlm-mlm-tlm-xnli15-1024-fintuned-on-xnli")
tokenizer = XLMTokenizer.from_pretrained("xlm-mlm-tlm-xnli15-1024-fintuned-on-xnli")
# 注意名字中不需要带paddle。调用后会自动从huggingface.co下载权重。
```

## 4. 开始使用

### 4.1 模型训练

命令已经配置完毕，可以直接运行。详细的参数配置可参考文件`args.py`
```bash
python train.py --output_dir facebook_xnli --pretrained_model_name_or_path xlm-mlm-tlm-xnli15-1024
```

```
04/12/2022 15:25:25 - INFO - __main__ - **********  Configuration Arguments **********
04/12/2022 15:25:25 - INFO - __main__ - adam_epsilon: 1e-08
04/12/2022 15:25:25 - INFO - __main__ - cache_dir: data_caches
04/12/2022 15:25:25 - INFO - __main__ - eval_batch_size: 32
04/12/2022 15:25:25 - INFO - __main__ - fp16: False
04/12/2022 15:25:25 - INFO - __main__ - gradient_accumulation_steps: 1
04/12/2022 15:25:25 - INFO - __main__ - language: en
04/12/2022 15:25:25 - INFO - __main__ - learning_rate: 2e-06
04/12/2022 15:25:25 - INFO - __main__ - log_dir: xnli_right_avg_new/logs
04/12/2022 15:25:25 - INFO - __main__ - logging_steps: 50
04/12/2022 15:25:25 - INFO - __main__ - max_length: 256
04/12/2022 15:25:25 - INFO - __main__ - max_train_steps: 0
04/12/2022 15:25:25 - INFO - __main__ - num_train_epochs: 250
04/12/2022 15:25:25 - INFO - __main__ - num_workers: 0
04/12/2022 15:25:25 - INFO - __main__ - optimizer: adam
04/12/2022 15:25:25 - INFO - __main__ - output_dir: xnli_right_avg_new
04/12/2022 15:25:25 - INFO - __main__ - overwrite_cache: False
04/12/2022 15:25:25 - INFO - __main__ - pretrained_model_name_or_path: xlm-mlm-tlm-xnli15-1024
04/12/2022 15:25:25 - INFO - __main__ - save_steps: 2500
04/12/2022 15:25:25 - INFO - __main__ - scale_loss: 32768
04/12/2022 15:25:25 - INFO - __main__ - seed: 12345
04/12/2022 15:25:25 - INFO - __main__ - sentences_per_epoch: 20000
04/12/2022 15:25:25 - INFO - __main__ - topk: 3
04/12/2022 15:25:25 - INFO - __main__ - train_batch_size: 8
04/12/2022 15:25:25 - INFO - __main__ - train_language: en
04/12/2022 15:25:25 - INFO - __main__ - writer_type: visualdl
04/12/2022 15:25:25 - INFO - __main__ - **************************************************
04/12/2022 15:25:44 - INFO - __main__ - ********** Running training **********
04/12/2022 15:25:44 - INFO - __main__ -   Num examples = None
04/12/2022 15:25:44 - INFO - __main__ -   Num Epochs = 250
04/12/2022 15:25:44 - INFO - __main__ -   Instantaneous train batch size = 8
04/12/2022 15:25:44 - INFO - __main__ -   Instantaneous eval batch size = 16
04/12/2022 15:25:44 - INFO - __main__ -   Total train batch size (w. accumulation) = 8
04/12/2022 15:25:44 - INFO - __main__ -   Gradient Accumulation steps = 1
04/12/2022 15:25:44 - INFO - __main__ -   Total optimization steps = 625000
04/12/2022 15:25:49 - INFO - __main__ - global_steps 50 loss: 7.46472330
04/12/2022 15:25:54 - INFO - __main__ - global_steps 100 loss: 5.02984764
04/12/2022 15:25:59 - INFO - __main__ - global_steps 150 loss: 4.60122725
04/12/2022 15:26:04 - INFO - __main__ - global_steps 200 loss: 4.87390495
04/12/2022 15:26:09 - INFO - __main__ - global_steps 250 loss: 4.09307530
04/12/2022 15:26:14 - INFO - __main__ - global_steps 300 loss: 3.93492581
04/12/2022 15:26:19 - INFO - __main__ - global_steps 350 loss: 4.10160220
04/12/2022 15:26:25 - INFO - __main__ - global_steps 400 loss: 3.10217629
04/12/2022 15:26:30 - INFO - __main__ - global_steps 450 loss: 3.54333124
04/12/2022 15:26:34 - INFO - __main__ - global_steps 500 loss: 3.30277427
04/12/2022 15:26:40 - INFO - __main__ - global_steps 550 loss: 3.00207584
04/12/2022 15:26:45 - INFO - __main__ - global_steps 600 loss: 3.02170183
04/12/2022 15:26:50 - INFO - __main__ - global_steps 650 loss: 2.98708332
04/12/2022 15:26:55 - INFO - __main__ - global_steps 700 loss: 2.62695073
04/12/2022 15:27:00 - INFO - __main__ - global_steps 750 loss: 2.70854754
04/12/2022 15:27:05 - INFO - __main__ - global_steps 800 loss: 2.70319111
04/12/2022 15:27:09 - INFO - __main__ - global_steps 850 loss: 2.61839391
04/12/2022 15:27:14 - INFO - __main__ - global_steps 900 loss: 2.72874597
04/12/2022 15:27:19 - INFO - __main__ - global_steps 950 loss: 2.50610820
04/12/2022 15:27:24 - INFO - __main__ - global_steps 1000 loss: 2.53542953
04/12/2022 15:27:29 - INFO - __main__ - global_steps 1050 loss: 2.25212755
04/12/2022 15:27:34 - INFO - __main__ - global_steps 1100 loss: 2.50348177
04/12/2022 15:27:39 - INFO - __main__ - global_steps 1150 loss: 2.14373983
04/12/2022 15:27:43 - INFO - __main__ - global_steps 1200 loss: 2.30000065
04/12/2022 15:27:48 - INFO - __main__ - global_steps 1250 loss: 2.27780871
04/12/2022 15:27:53 - INFO - __main__ - global_steps 1300 loss: 2.27487551
04/12/2022 15:27:58 - INFO - __main__ - global_steps 1350 loss: 2.31865400
04/12/2022 15:28:03 - INFO - __main__ - global_steps 1400 loss: 2.54873763
04/12/2022 15:28:08 - INFO - __main__ - global_steps 1450 loss: 2.16360084
04/12/2022 15:28:12 - INFO - __main__ - global_steps 1500 loss: 2.09571365
04/12/2022 15:28:17 - INFO - __main__ - global_steps 1550 loss: 2.22257620
04/12/2022 15:28:22 - INFO - __main__ - global_steps 1600 loss: 1.93775404
04/12/2022 15:28:27 - INFO - __main__ - global_steps 1650 loss: 2.05469335
04/12/2022 15:28:31 - INFO - __main__ - global_steps 1700 loss: 2.12268977
04/12/2022 15:28:36 - INFO - __main__ - global_steps 1750 loss: 1.98807700
04/12/2022 15:28:41 - INFO - __main__ - global_steps 1800 loss: 1.89246038
04/12/2022 15:28:46 - INFO - __main__ - global_steps 1850 loss: 1.80554641
04/12/2022 15:28:50 - INFO - __main__ - global_steps 1900 loss: 2.01120813
04/12/2022 15:28:55 - INFO - __main__ - global_steps 1950 loss: 1.91710269
04/12/2022 15:29:00 - INFO - __main__ - global_steps 2000 loss: 1.78628310
04/12/2022 15:29:04 - INFO - __main__ - global_steps 2050 loss: 1.95788345
04/12/2022 15:29:09 - INFO - __main__ - global_steps 2100 loss: 1.71753185
04/12/2022 15:29:14 - INFO - __main__ - global_steps 2150 loss: 1.72524106
04/12/2022 15:29:18 - INFO - __main__ - global_steps 2200 loss: 1.98808228
04/12/2022 15:29:23 - INFO - __main__ - global_steps 2250 loss: 1.72786968
04/12/2022 15:29:28 - INFO - __main__ - global_steps 2300 loss: 1.74996327
04/12/2022 15:29:32 - INFO - __main__ - global_steps 2350 loss: 1.69286092
04/12/2022 15:29:37 - INFO - __main__ - global_steps 2400 loss: 1.64929492
04/12/2022 15:29:42 - INFO - __main__ - global_steps 2450 loss: 1.78267193
04/12/2022 15:29:46 - INFO - __main__ - global_steps 2500 loss: 1.74447792
04/12/2022 15:29:46 - INFO - __main__ - ********** Running evaluating **********
04/12/2022 15:29:51 - INFO - __main__ - ##########  val_ar_acc 0.3389558232931727 ##########
04/12/2022 15:29:55 - INFO - __main__ - ##########  val_bg_acc 0.41244979919678715 ##########
04/12/2022 15:30:00 - INFO - __main__ - ##########  val_de_acc 0.3481927710843373 ##########
04/12/2022 15:30:05 - INFO - __main__ - ##########  val_el_acc 0.46586345381526106 ##########
04/12/2022 15:30:09 - INFO - __main__ - ##########  val_en_acc 0.5333333333333333 ##########
04/12/2022 15:30:13 - INFO - __main__ - ##########  val_es_acc 0.4682730923694779 ##########
04/12/2022 15:30:18 - INFO - __main__ - ##########  val_fr_acc 0.38072289156626504 ##########
04/12/2022 15:30:23 - INFO - __main__ - ##########  val_hi_acc 0.3674698795180723 ##########
04/12/2022 15:30:28 - INFO - __main__ - ##########  val_ru_acc 0.385140562248996 ##########
04/12/2022 15:30:32 - INFO - __main__ - ##########  val_sw_acc 0.3369477911646586 ##########
04/12/2022 15:30:36 - INFO - __main__ - ##########  val_th_acc 0.4827309236947791 ##########
04/12/2022 15:30:40 - INFO - __main__ - ##########  val_tr_acc 0.3863453815261044 ##########
04/12/2022 15:30:45 - INFO - __main__ - ##########  val_ur_acc 0.3373493975903614 ##########
04/12/2022 15:30:49 - INFO - __main__ - ##########  val_vi_acc 0.3333333333333333 ##########
04/12/2022 15:30:54 - INFO - __main__ - ##########  val_zh_acc 0.3337349397590361 ##########
04/12/2022 15:31:02 - INFO - __main__ - ##########  test_ar_acc 0.3379241516966068 ##########
04/12/2022 15:31:12 - INFO - __main__ - ##########  test_bg_acc 0.4119760479041916 ##########
04/12/2022 15:31:21 - INFO - __main__ - ##########  test_de_acc 0.35129740518962077 ##########
04/12/2022 15:31:31 - INFO - __main__ - ##########  test_el_acc 0.4756487025948104 ##########
04/12/2022 15:31:40 - INFO - __main__ - ##########  test_en_acc 0.5467065868263473 ##########
04/12/2022 15:31:49 - INFO - __main__ - ##########  test_es_acc 0.4750499001996008 ##########
04/12/2022 15:31:58 - INFO - __main__ - ##########  test_fr_acc 0.3870259481037924 ##########
04/12/2022 15:32:09 - INFO - __main__ - ##########  test_hi_acc 0.37025948103792417 ##########
04/12/2022 15:32:19 - INFO - __main__ - ##########  test_ru_acc 0.37984031936127743 ##########
04/12/2022 15:32:28 - INFO - __main__ - ##########  test_sw_acc 0.3401197604790419 ##########
04/12/2022 15:32:37 - INFO - __main__ - ##########  test_th_acc 0.4820359281437126 ##########
04/12/2022 15:32:47 - INFO - __main__ - ##########  test_tr_acc 0.3908183632734531 ##########
04/12/2022 15:32:57 - INFO - __main__ - ##########  test_ur_acc 0.33672654690618764 ##########
04/12/2022 15:33:06 - INFO - __main__ - ##########  test_vi_acc 0.3333333333333333 ##########
04/12/2022 15:33:15 - INFO - __main__ - ##########  test_zh_acc 0.3333333333333333 ##########
04/12/2022 15:33:15 - INFO - __main__ -   val_ar_acc = 0.3389558232931727
04/12/2022 15:33:15 - INFO - __main__ -   val_bg_acc = 0.41244979919678715
04/12/2022 15:33:15 - INFO - __main__ -   val_de_acc = 0.3481927710843373
04/12/2022 15:33:15 - INFO - __main__ -   val_el_acc = 0.46586345381526106
04/12/2022 15:33:15 - INFO - __main__ -   val_en_acc = 0.5333333333333333
04/12/2022 15:33:15 - INFO - __main__ -   val_es_acc = 0.4682730923694779
04/12/2022 15:33:15 - INFO - __main__ -   val_fr_acc = 0.38072289156626504
04/12/2022 15:33:15 - INFO - __main__ -   val_hi_acc = 0.3674698795180723
04/12/2022 15:33:15 - INFO - __main__ -   val_ru_acc = 0.385140562248996
04/12/2022 15:33:15 - INFO - __main__ -   val_sw_acc = 0.3369477911646586
04/12/2022 15:33:15 - INFO - __main__ -   val_th_acc = 0.4827309236947791
04/12/2022 15:33:15 - INFO - __main__ -   val_tr_acc = 0.3863453815261044
04/12/2022 15:33:15 - INFO - __main__ -   val_ur_acc = 0.3373493975903614
04/12/2022 15:33:15 - INFO - __main__ -   val_vi_acc = 0.3333333333333333
04/12/2022 15:33:15 - INFO - __main__ -   val_zh_acc = 0.3337349397590361
04/12/2022 15:33:15 - INFO - __main__ -   val_avg_acc = 0.39405622489959846
04/12/2022 15:33:15 - INFO - __main__ -   test_ar_acc = 0.3379241516966068
04/12/2022 15:33:15 - INFO - __main__ -   test_bg_acc = 0.4119760479041916
04/12/2022 15:33:15 - INFO - __main__ -   test_de_acc = 0.35129740518962077
04/12/2022 15:33:15 - INFO - __main__ -   test_el_acc = 0.4756487025948104
04/12/2022 15:33:15 - INFO - __main__ -   test_en_acc = 0.5467065868263473
04/12/2022 15:33:15 - INFO - __main__ -   test_es_acc = 0.4750499001996008
04/12/2022 15:33:15 - INFO - __main__ -   test_fr_acc = 0.3870259481037924
04/12/2022 15:33:15 - INFO - __main__ -   test_hi_acc = 0.37025948103792417
04/12/2022 15:33:15 - INFO - __main__ -   test_ru_acc = 0.37984031936127743
04/12/2022 15:33:15 - INFO - __main__ -   test_sw_acc = 0.3401197604790419
04/12/2022 15:33:15 - INFO - __main__ -   test_th_acc = 0.4820359281437126
04/12/2022 15:33:15 - INFO - __main__ -   test_tr_acc = 0.3908183632734531
04/12/2022 15:33:15 - INFO - __main__ -   test_ur_acc = 0.33672654690618764
04/12/2022 15:33:15 - INFO - __main__ -   test_vi_acc = 0.3333333333333333
04/12/2022 15:33:15 - INFO - __main__ -   test_zh_acc = 0.3333333333333333
04/12/2022 15:33:15 - INFO - __main__ -   test_avg_acc = 0.3968063872255489
04/12/2022 15:33:15 - INFO - __main__ - ########## Step 2500 val_avg_acc 0.39405622489959846 test_avg_acc 0.3968063872255489 ##########
```
Tips: 训练日志保存在`logs`文件夹，里面有训练过程的日志，可以使用visualdl打开查看训练过程中指标的变化。


### 4.2 模型评估

可直接从huggingface.co加载微调好的权重。
```bash
python eval.py --output_dir eval_output --pretrained_model_name_or_path xlm-mlm-tlm-xnli15-1024-fintuned-on-xnli
```

```
04/12/2022 22:43:41 - INFO - __main__ - ********** Running evaluating **********
04/12/2022 22:43:46 - INFO - __main__ - ##########  val_ar_acc 0.7172690763052209 ##########
04/12/2022 22:43:50 - INFO - __main__ - ##########  val_bg_acc 0.7606425702811245 ##########
04/12/2022 22:43:55 - INFO - __main__ - ##########  val_de_acc 0.7767068273092369 ##########
04/12/2022 22:43:59 - INFO - __main__ - ##########  val_el_acc 0.7522088353413655 ##########
04/12/2022 22:44:04 - INFO - __main__ - ##########  val_en_acc 0.8437751004016064 ##########
04/12/2022 22:44:08 - INFO - __main__ - ##########  val_es_acc 0.7891566265060241 ##########
04/12/2022 22:44:13 - INFO - __main__ - ##########  val_fr_acc 0.7811244979919679 ##########
04/12/2022 22:44:18 - INFO - __main__ - ##########  val_hi_acc 0.6899598393574298 ##########
04/12/2022 22:44:22 - INFO - __main__ - ##########  val_ru_acc 0.7429718875502008 ##########
04/12/2022 22:44:27 - INFO - __main__ - ##########  val_sw_acc 0.6714859437751004 ##########
04/12/2022 22:44:31 - INFO - __main__ - ##########  val_th_acc 0.706425702811245 ##########
04/12/2022 22:44:36 - INFO - __main__ - ##########  val_tr_acc 0.7208835341365462 ##########
04/12/2022 22:44:40 - INFO - __main__ - ##########  val_ur_acc 0.6461847389558233 ##########
04/12/2022 22:44:45 - INFO - __main__ - ##########  val_vi_acc 0.7397590361445783 ##########
04/12/2022 22:44:49 - INFO - __main__ - ##########  val_zh_acc 0.751004016064257 ##########
04/12/2022 22:44:58 - INFO - __main__ - ##########  test_ar_acc 0.7381237524950099 ##########
04/12/2022 22:45:08 - INFO - __main__ - ##########  test_bg_acc 0.7758483033932135 ##########
04/12/2022 22:45:17 - INFO - __main__ - ##########  test_de_acc 0.7686626746506986 ##########
04/12/2022 22:45:26 - INFO - __main__ - ##########  test_el_acc 0.7676646706586826 ##########
04/12/2022 22:45:35 - INFO - __main__ - ##########  test_en_acc 0.846307385229541 ##########
04/12/2022 22:45:45 - INFO - __main__ - ##########  test_es_acc 0.7978043912175649 ##########
04/12/2022 22:45:54 - INFO - __main__ - ##########  test_fr_acc 0.7924151696606786 ##########
04/12/2022 22:46:04 - INFO - __main__ - ##########  test_hi_acc 0.6880239520958084 ##########
04/12/2022 22:46:13 - INFO - __main__ - ##########  test_ru_acc 0.7618762475049901 ##########
04/12/2022 22:46:22 - INFO - __main__ - ##########  test_sw_acc 0.691816367265469 ##########
04/12/2022 22:46:31 - INFO - __main__ - ##########  test_th_acc 0.7109780439121757 ##########
04/12/2022 22:46:40 - INFO - __main__ - ##########  test_tr_acc 0.7173652694610778 ##########
04/12/2022 22:46:49 - INFO - __main__ - ##########  test_ur_acc 0.6584830339321357 ##########
04/12/2022 22:46:59 - INFO - __main__ - ##########  test_vi_acc 0.7447105788423154 ##########
04/12/2022 22:47:08 - INFO - __main__ - ##########  test_zh_acc 0.7481037924151697 ##########
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

### 4.3 模型预测

建议进入`test_tipc`文件夹查看详细信息。

## 5. 模型推理部署

建议进入`test_tipc`文件夹查看详细信息。

## 6. 自动化测试脚本

建议进入`test_tipc`文件夹查看详细信息。

## 7. LICENSE

本项目的发布受[Apache 2.0 license](./LICENSE)许可认证。

## 8. 参考链接与文献

- https://github.com/facebookresearch/XLM
- https://github.com/huggingface/transformers/tree/main/src/transformers/models/xlm
- https://github.com/PaddlePaddle/PaddleNLP
- https://arxiv.org/pdf/1901.07291.pdf
