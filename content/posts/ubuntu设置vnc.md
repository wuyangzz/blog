---
title: "Ubuntu设置vnc"
author: "wuyangzz"
tags: ["ubuntu"]
categories: ["ubuntu"]
date: 2021-03-09T13:43:38+08:00
---

# 安装配置软件
- ## VNC的安装与配置
安装之前先输入（获取最新套件的信息）

```shell 
apt-get update
```
输入以下命令安装VNC，安装过程中需要输入Y来确认
```sehll
apt-get install vnc4server
```
启动VNC（第一次启动需要设置密码）
```shell
vncserver 
```

# 设置vncservgnome 桌面环境安装与配置（可直接跳至第3步）

安装x－windows的基础

```shell
sudo apt-get install x-window-system-core
 ```

安装登录管理器

```shell
sudo apt-get install gdm
```

安装Ubuntu的桌面
```shell

sudo apt-get install ubuntu-desktop
```

安装gnome配套软件

```shell
sudo apt-get install gnome-panel gnome-settings-daemon metacity nautilus gnome-terminal
```
修改VNC配置文件

```shell
sudo vim ~/.vnc/xstartup
```

修改为：
```shell
#!/bin/sh
# Uncomment the following two lines for normal desktop:
export XKL_XMODMAP_DISABLE=1
 unset SESSION_MANAGER
# exec /etc/X11/xinit/xinitrc
unset DBUS_SESSION_BUS_ADDRESS
gnome-panel &
gnmoe-settings-daemon &
metacity &
nautilus &
gnome-terminal &
```

杀掉原桌面进程，输入命令（其中的:1是桌面号）：

```shell
vncserver -kill :1
```
输入以下命令生成新的会话：
```shell
vncserver :1
```

ubuntu卸载gnome桌面（可直接跳至第3步）
之前安装好了ubuntu18.04，本来想装个gnome shell来美化一下桌面，结果出现了开机黑屏（灰屏）的现象，经网上查询发现是显卡驱动在gnome3的环境下产生了不兼容，具体解决方法我还没找到，情急之下只能先卸载掉gnome桌面环境。

卸载掉gnome-shell主程序
```shell
sudo apt-get remove gnome-shell\
```

卸载掉gnome
```shell
sudo apt-get remove gnome
```

卸载不需要的依赖关系
```shell
sudo apt-get autoremove
```

彻底卸载删除gnome的相关配置文件
```shell
sudo apt-get purge gnome
```
清理安装gnome时候留下的缓存程序软件包
```shell
sudo apt-get autoclean
sudo apt-get clean
```
ubuntu运行VNC Server无桌面时的解决方案
配置vnc server实在是一个特别诡异的事，我在不同的ubuntu机器上配置服务时，总是遇到千奇百怪的问题，大部分情况下比较顺利，将~/.vnc/xstartup最后一句```x-window-manager&```替换为```gnome-session&```就能顺利地出现桌面，而有些则不行，需要改为```gnome-session --session=ubuntu-2d&```

# 问题
而今天遇到的ubuntu 18.04，则死活不行，用realvnc viewer连接之后，只有灰灰的一个背景，没有桌面，没有terminal。
使用xfce4桌面解决，选择些方案一是因为xfce相对较小，gnome-session死活不行的情况下，再将一个kde未免太过兴师动众，xfce我使用过一段时间，是一个相当轻量级的GUI环境，清爽易用，功能一点不含糊。决定之后，一次尝试即成功。

安装xfce
```shell
sudo apt-get install gnome-core xfce4
```
配置
编辑~/.vnc/xstartup文件：
```shell
#!/bin/sh
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
startxfce4 &
[ -x /etc/vnc/xstartup ] && exec /etc/vnc/xstartup
[ -r $HOME/.Xresources ] && xrdb $HOME/.Xresources
xsetroot -solid grey
vncconfig -iconic &
```

输入以下命令生成新的会话：

```shell
vncserver :1
```
本地使用VNC连接
本地安装VNC后，使用ip地址:1（其中的:1是桌面号）的方式连接
输入之前设置的VNC密码后点击连接
# 设置虚拟分辨率启动
```shell 
 vnc4server -geometry 1920x1440
```