---
title: "GPU租用手册"
author: "wuyangzz"
tags: ["GPU"]
categories: ["实验室GPU"]
date: 2021-02-22T14:17:51+08:00
---


# 一、服务器简介

## 概述

实验室利用两台ESC 4000G2 服务器 4张3090显卡搭建GPU服务器。但是因为3090非专业显卡。不支持vGPU功能。如果大家都直接操作宿主主机，直接在宿主主机上配置自己的开发环境将会发生不可避免的冲突。最后经过实际考虑，利用Docker进行合理的系统资源的搭配。使用 Docker 把服务器容器化，每个人都直接登录自己的容器，所有开发都在自己的容器内完成，这样就避免了冲突。并且，Docker 容器的额外开销小得可以忽略不计，所以也不会影响服务器性能。

- 一个docker镜像就可以看作是一个操作系统。在docker上面进行的操作不会影响主机本生的环境。
- 虚拟容器采用docker方式实现，为了能在docker中可以使用GPU。采用nvidia-docker进行gpu的加载。nvidia-docker 是专门为需要访问显卡资源的容器量身定制的，它对原始的 Docker 命令作了封装，只要使用 nvidia-docker run 命令运行容器，容器就可以访问主机显卡设备（只要主机安装了显卡驱动）。
- 如果要在docker中使用显卡。现在NVIDIA给出的解决方案中必须使用linux系统。
- 可以在docker中加载基础的镜像，然后将22端口映射出来。就可以直接使用主机ip加映射的端口来访问和使用docker容器。
- 可以使用web界面如Shipyard等来对docker进行GUI管理
- NVIDIA有官方的Docker目录网站NGC，NGC为AI，机器学习和HPC提供了GPU加速容器的综合中心，这些容器已优化，测试并可以在本地和云中受支持的NVIDIA GPU上运行。此外，它提供了可以轻松集成到现有工作流程中的预训练模型，模型脚本和行业解决方案。
- [NGC网站](https://ngc.nvidia.com/catalog/containers?orderBy=modifiedDESC&pageNumber=0&query=&quickFilter=containers&filters=)镜像中包含很多包，例如TensorFlow，PyTorch，MXNet，NVIDIA TensorRT™，RAPIDS等，并且有各个版本的组合可以下载。更新也非常快。
- 
## 服务器基本配置

| 备注 |	HPC1	|  HPC2 |  
| :----:    | :----:   | :----: |
|CPU	|Intel(R) Xeon(R) CPU E5-2620  2.00GHz|	Intel(R) Xeon(R) CPU E5-2620  2.00GHz|
|内存|	64 G| （8*8G）|	128 G （4*32G）|
|硬盘|	3T|	2T|
|显卡|	技嘉RTX 3090 Turbo*2|	技嘉RTX 3090 Turbo*2|
|IP	|172.23.253.104|	172.23.253.113|
|Driver Version| 460.32.03|460.32.03|
|CUDA| CUDA11.2 |CUDA11.2|



# 二、知识准备

## 2.1 下载Xshell、Xftp，并了解如何使用

## 2.2 了解什么是Docker、什么是容器

## 2.3 了解基本的ubuntu使用命令

## 2.4 了解Jupyter lab使用命令



# 三、基础环境

## 3.1 直接提供容器

### 3.1.1 PyTorch容器

PyTorch是具有Python前端的GPU加速张量计算框架。使用常见的Python库（例如NumPy，SciPy和Cython）可以轻松扩展功能。利用功能性和神经网络层级的基于磁带的系统可以完成自动区分。作为深度学习框架，此功能带来了高度的灵活性和速度，并提供了类似于NumPy的加速功能。本容器基于[NGC PyTorch Release 20.12](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_20-12.html#rel_20-12)。
此容器映像包含/ opt / pytorch中PyTorch版本的完整源。它是预构建的，并安装在容器映像中的Conda默认环境（/opt/conda/lib/python3.6/site-packages/torch/）中。

基本软件环境

* Ubuntu 18.04 including Python 3.6 environment
* Pytorch 1.8.0a0 + 1606899
* NVIDIA CUDA 11.1.0 including cuBLAS 11.2.1
* NVIDIA cuDNN 8.0.4
* APEX
* MLNX_OFED 5.1
* OpenMPI 4.0.5
* TensorBoard 1.15.0+nv20.11
* Nsight Compute 2020.2.0.18
* Nsight Systems 2020.3.4.32
* TensorRT 7.2.1
* Tensor Core optimized examples
* Jupyter and JupyterLab:
* ssh server
* 其他软件包，用户自行安装。

### 3.1.2 Tensorflow容器

TensorFlow是用于使用数据流图进行数值计算的开源软件库。图中的节点表示数学运算，而图的边缘表示在它们之间流动的多维数据数组（张量）。这种灵活的体系结构允许您将计算部署到台式机，服务器或移动设备中的一个或多个CPU或GPU上，而无需重写代码。本容器基于[NGC TensorFlow Release 20.12](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel_20-12.html#rel_20-12)。
此容器包括的NVIDIA版本的完整源TensorFlow在/ opt / tensorflow。它是预先构建的，并作为系统Python模块安装。

其他基本软件环境
* Ubuntu 20.04
* Python 3.8
* Tensorflow 1.15.4 and 2.3.1
* NVIDIA CUDA 11.1.1 including cuBLAS 11.3.0
* NVIDIA cuDNN 8.0.5
* Horovod 0.20.2
* OpenMPI 4.0.5
* TensorBoard
* MLNX_OFED 5.1
* TensorRT 7.2.2
* DALI 0.28
* Nsight Compute 2020.2.1.8
* Nsight Systems 2020.3.4.32
* XLA-Lite
* Tensor Core optimized examples
* JupyterLab 1.2.14
* ssh server
* 其他软件包，用户自行安装。

## 3.2 NGC官网自定义容器

### 3.2.1 注册并登录[NGC网站](https://ngc.nvidia.com/catalog/containers?orderBy=modifiedDESC&pageNumber=0&query=&quickFilter=containers&filters=)

### 3.2.2 打开[NGC网站](https://ngc.nvidia.com/catalog/containers?orderBy=modifiedDESC&pageNumber=0&query=&quickFilter=containers&filters=)，其中可以浏览自己所需要的容器

![image-20201206195516130](https://raw.githubusercontent.com/wycyz1/blog_image/main/20201206195516.png)

### 3.2.3 自己根据自己的基础环境 如Tensorflow、Pytorch进行搜索。

进入其中。里面有该容器的参考文档请仔细阅读（参考文档中有基本环境的配置）。只需要将其Pull记住并告诉我。我将做一些基本的配置后即可使用。

![image-20201207181045933](https://raw.githubusercontent.com/wycyz1/blog_image/main/20201207181251.png)

# 四、收费情况

为了便于计算，将只对GPU使用数量和使用时长进行计费。

|序号|	套餐	|GPU数量|	计费方式|	单价|	均价|
| :----: | :----: | :----: | :----: | :----: |:----: |
| 1 |	套餐一|	单张3090|	按小时计费|	2元/小时 	|2元/小时|
|2|	套餐二 |	单张3090 |	按天计费 |	35元/每天 |	1.45/小时|
|3|	套餐三 |	单张3090 |	按周计费 |	220元/每周 |	1.31元/小时 |
|4|	套餐四 |	单张3090 |	按月计费 |	750元/每月 |	1.12元/小时 |
|5|	套餐五 |	两张3090 |	按小时计费 |	3.6元/小时 |	3.6元/小时 |
|6|	套餐六 |	两张3090 |	按天计费 |	75元/每天 |	3.125/小时 |
|7|	套餐七 |	两张3090 |	按周计费 |	420元/每周 |	2.5元/小时 |
|8|	套餐八 |	两张3090 |	按月计费 |	1400元/每月	 |2.08元/小时 |

**其他收费**
|序号| 项目 |费用 |
| :----: | :----: | :----: |
|1|更换镜像或者端口|30元/次|

# 五、登录方式

## 5.1 告知基础环境、需要开放的端口、映射文件夹以及用途

我们一般会提供一个使用jupyter的**8888**端口和一个使用xshell和xftp的**22**端口。会将容器中的**workspace**目录映射到Host主机，以免文件丢失。如果有特殊需求，需要提前告知。

## 5.2 填写附件1申请服务器资源

## 5.3 SSH登录方法

用户名： root
初始密码： root
ip：给定
端口：给定
登录：``` ssh root@<ip> -p <端口>```
登录后立即修改密码：运行 passwd
## 5.4 数据上传和下载

使用常见的FTP软件xftp/MobaXterm进行数据的上传和下载
主机名(Host): 给定
端口(Port) : 给定
传输协议(Protocal): SFTP
用户名: root
密码: 你的ssh登录密码

**数据存放建议**
**所有个人数据建议存放在 /workspace下**

## 5.5 Jupyter

Jupyter Notebook端口：给定
Jupyter Notebook默认密码：root
设置并开启Jupyter Notebook的方法
第一步：在服务器打开终端，输入``` jupyter notebook --generate-config ``` 生成配置文件
第二步：```shell jupyter notebook password ```   *# 按提示，输入密码*
第三步：后台运行
```shell
nohup jupyter-lab --ip 0.0.0.0 --port 8888 --allow-root > jupyter.log 2>&1 &  
```
第五步：在自己的电脑即可登录jupyter notebook:
http://给定IP: <Jupyter端口>

# 六、连接容器

## 6.1 使用Xshell新建链接

![image-20210102205318604](https://raw.githubusercontent.com/wycyz1/blog_image/main/20210102205318.png)

## 6.2 链接配置

IP和端口配置（IP和端口都会告知）

<img src="https://raw.githubusercontent.com/wycyz1/blog_image/main/20210102205437.png" alt="image-20210102205437002" style="zoom:50%;" />

输入账户名和密码（用户名和密码默认为root）

<img src="https://raw.githubusercontent.com/wycyz1/blog_image/main/20210102205554.png" alt="image-20210102205553973" style="zoom:50%;" />


# 七、问题咨询

## 7.1 我能解决的问题

	- 容器连接不上（操作无误的情况下）
	- 需要开放其他额外端口
	- 需要几张显卡
	- 需要使用时间
	- 容器需要重启

## 7.2 需要自己解决的问题

	- Ubuntu怎么使用
	- Jupyter怎么使用
	- 如何选择适合自己的镜像
	- 怎么上传、下载文件
	- 镜像里面有什么环境（NGC官网里面有详细的指导文档）
	- 需要更改容器环境