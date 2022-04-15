# Inference 推理


# 目录

- [1. 简介]()
- [2. 推理过程]()
    - [2.1 准备推理环境]()
    - [2.2 模型动转静导出]()
    - [2.3 模型推理]()
- [3. FAQ]()


## 1. 简介

Paddle Inference 是飞桨的原生推理库， 作用于服务器端和云端，提供高性能的推理能力。相比于直接基于预训练模型进行预测，Paddle Inference可使用MKLDNN、CUDNN、TensorRT进行预测加速，从而实现更优的推理性能。

本文档主要基于Paddle Inference的XLM模型推理。

更多关于Paddle Inference推理引擎的介绍，可以参考[Paddle Inference官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/inference_cn.html)。


## 2. 推理过程

### 2.1 准备推理环境

安装好PaddlePaddle即可体验Paddle Inference部署能力。


### 2.2 模型动转静导出

使用下面的命令完成`XLM`模型的动转静导出。
其中`xlm-mlm-tlm-xnli15-1024-fintuned-on-xnli`是训练好的最优权重`test avg acc = 74.7`，为了便于加载已经上传至`huggingface.co`.
```bash
python tools/export_model.py --model_path=xlm-mlm-tlm-xnli15-1024-fintuned-on-xnli --save_inference_dir=./xlm_infer
```

最终在`xlm_infer/`文件夹下会生成下面的3个文件。

```
xlm_infer
     |----inference.pdiparams     : 模型参数文件
     |----inference.pdmodel       : 模型结构文件
     |----inference.pdiparams.info: 模型参数信息文件
```

### 2.3 模型推理


```bash
python deploy/inference_python/infer.py --model_dir=./xlm_infer/
```

对于下面的文本进行预测(其中<sep>标记是为区分两个句子。)

`You don't have to stay there.<sep>You can leave.`

在终端中输出结果如下。

```
text: You don't have to stay there.<sep>You can leave., label_id: 0, prob: 0.9517736434936523
```

表示预测的类别ID是`0`，置信度为`0.9517736434936523`，该结果与基于训练引擎的结果完全一致。


## 3. FAQ
