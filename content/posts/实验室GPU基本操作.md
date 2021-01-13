---
title: "实验室GPU基本操作"
date: 2021-01-10T10:28:19+08:00
categories: ["实验室GPU"]
tags: ["GPU"]
---

# 一、前期准备

## 1. 下载Xshell、Xftp，并了解如何使用

## 2. 了解什么是Docker、什么是容器

## 3. 了解基本的ubuntu使用命令

## 4. 了解Jupyter lab使用命令

# 二、寻找自己所需要的基础环境

## 1. 注册并登录[NGC网站](https://ngc.nvidia.com/catalog/containers?orderBy=modifiedDESC&pageNumber=0&query=&quickFilter=containers&filters=)

## 2. 打开[NGC网站](https://ngc.nvidia.com/catalog/containers?orderBy=modifiedDESC&pageNumber=0&query=&quickFilter=containers&filters=)，其中可以浏览自己所需要的容器

![image-20201206195516130](https://raw.githubusercontent.com/wycyz1/blog_image/main/20201206195516.png)

## 3. 自己根据自己的基础环境 如Tensorflow Pytorch进行搜索。

	并进入其中。里面有该容器的参考文档请仔细阅读（参考文档中有基本环境的配置）。只需要将其Pull记住并告诉我。

![image-20201207181045933](https://raw.githubusercontent.com/wycyz1/blog_image/main/20201207181251.png)

# 三、告知需要开放的端口和映射文件夹以及用途

我们一般会提供一个使用jupyter的**8888**端口和一个使用xshell和xftp的**22**端口。会将容器中的**workspace**目录映射到Host主机，以免文件丢失。如果有特殊需求，需要提前告知。

# 四、连接容器

## 1. 使用Xshell新建链接

![image-20210102205318604](https://raw.githubusercontent.com/wycyz1/blog_image/main/20210102205318.png)

## 2. 链接配置

IP和端口配置（IP和端口都会告知）

<img src="https://raw.githubusercontent.com/wycyz1/blog_image/main/20210102205437.png" alt="image-20210102205437002" style="zoom:50%;" />

输入账户名和密码（用户名和密码默认为root）

<img src="https://raw.githubusercontent.com/wycyz1/blog_image/main/20210102205554.png" alt="image-20210102205553973" style="zoom:50%;" />


# 五、启动Jupyter lab

```shell
nohup jupyter-lab --ip 0.0.0.0 --port 8888 --allow-root > jupyter.log 2>&1 &  
```



![image-20210102205804952](https://raw.githubusercontent.com/wycyz1/blog_image/main/20210102210010.png)

# 六、连接Jupyter lab

当在容器中成功启动Jupyter lab后就可以在浏览器中输入给定的IP和端口对jupyter lab进行连接，连接密码默认为root。

建议使用时，若无特殊要求，请将所有自己的文件放置在workspace目录下，以免丢失。

# 七、问题咨询

## 1. 我能解决的问题

	- 容器连接不上（操作无误的情况下）
	- Jupyter端口打开不了
	- 需要开放其他额外端口
	- 需要几张显卡
	- 需要使用时间
	- 容器需要重启

## 2. 需要自己解决的问题

	- Ubuntu怎么使用
	- Jupyter怎么使用
	- 如何选择适合自己的镜像
	- 怎么上传、下载文件
	- 镜像里面有什么环境（NGC官网里面有详细的指导文档）
	- 需要更改容器环境

	