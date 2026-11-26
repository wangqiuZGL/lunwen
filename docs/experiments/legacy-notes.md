# HIVG--base模型

## unc：

### 作者训练好的权重：87.32（↓0.61）  89.86（↓0.6） 83.27（↓1%）

```json
{
"test_model:":"/data/students/zhangyingjian/testDemo/HiVG/data/HiVG_ckpt/finetuning_base/unc/best_checkpoint.pth",
"val_set_accuracy":0.8671774044674174
}


{
"test_model:":"/data/students/zhangyingjian/testDemo/HiVG/data/HiVG_ckpt/finetuning_base/unc/best_checkpoint.pth",
"testA_set_accuracy":0.8921880523153057
}

{"test_model:": "/data/students/zhangyingjian/testDemo/HiVG/data/HiVG_ckpt/finetuning_base/unc/best_checkpoint.pth", "testB_set_accuracy": 0.8222135007849294}

```





### 重新训练       87.32（↓0.63）  89.86（持平） 83.27（↓0.6%）

```json
{
    "test_model:": "/data/students/zhangyingjian/testDemo/HiVG/HiVG_output/unc_stage3/best_checkpoint.pth",
    "val_set_accuracy": 0.8669004984308658
}

{
    "test_model:": "/data/students/zhangyingjian/testDemo/HiVG/HiVG_output/unc_stage3/best_checkpoint.pth",
    "testA_set_accuracy": 0.8985507246376812
}

{
    "test_model:": "/data/students/zhangyingjian/testDemo/HiVG/HiVG_output/unc_stage3/best_checkpoint.pth",
    "testB_set_accuracy": 0.8267268445839875
}

```





## ReferIt

### 作者训练好的权重：75.22（↑0.1）

```json

```



### 重新训练：75.22

```json

```



# oneref

![image-20251221142429700](E:\myproject\mycode\笔记\assets\image-20251221142429700.png)

## unc-rec

作者：88.75 90.95 85.34

复现效果： 90.08  91.64  86.36



加入diff模块：89.20 90.94 85.84

{"epoch": 19, "train_lr": 1.846748910729252e-07, "train_loss": 1.9960856410501127, "train_loss_bbox": 0.050437232665649645, "train_loss_giou": 0.1467800945307229, "train_loss_mrm_focal": 0.3690595362839669, "train_loss_mrm_dice": 1.4298087776207393, "validation_box_miou": 0.8029683750230755, "validation_box_accu": 0.8906221155621192, "validation_accu": 0.8906221155621192, "n_parameters": 228458886}



## unc+

作者效果： 80.43 86.46 74.26



加入语法模块：

复现效果：  82.40 86.93 75.99





## unc-g

作者： 83.68 83.52

加入句法模块：

82. 557    83.71



在训练10轮

val:82.924

 test:84.16







## ReferIt

作者： 77.17

复现效果： 78.60



加入句法模块【训练20轮】

val : 80.229

test : 78.377

【训练23轮】

val:80.24

test:78.43



【训练30轮】

val：80.69

78.51



## flicker30



作者：83.61



【加入句法模块】

val：82.028

test: 82.7510
