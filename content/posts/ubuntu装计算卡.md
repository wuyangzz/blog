---
title: "Ubuntu装计算卡"
author: "wuyangzz"
tags: ["专业卡"]
categories: ["ubuntu"]
date: 2021-03-11T11:29:01+08:00
---
# 背景
实验室台式机电脑本身有intel的集成显卡，但是在装ubuntu桌面版的时候，安装好专业卡K20C（无输出端口）的驱动后 无法进入桌面。
![lADPD4d8sEdmYZrND8DNC9A_3024_4032](https://raw.githubusercontent.com/wuyangzz/blog_image/main/lADPD4d8sEdmYZrND8DNC9A_3024_4032.jpg)
但是进入恢复模式卸载 NVIDIA显卡驱动后便可以正常进入系统。
# 解决方案
- ## 强制重启时进入UBUNTU高级选项（advanced options)
![20210311115317](https://raw.githubusercontent.com/wuyangzz/blog_image/main/20210311115317.png)
- ## 选择root模式进入
![20210311115352](https://raw.githubusercontent.com/wuyangzz/blog_image/main/20210311115352.png)
- ## 设置默认显卡为intel
```shell
sudo prime-select query # 查看当前显卡
sudo prime-select intel # 设置Intel显卡
sudo prime-select nvidia # 设置NVIDIA显卡
```
- ## 重启
- ## 开启nvidia显卡
正常重启后可以进入系统，但是还是没有调用NVIDIA显卡
可以直接在终端输入命令(前提是你已经安装好了NVIDIA的驱动)：
```shell
nvidia-settings
```