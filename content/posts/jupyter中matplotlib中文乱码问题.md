---
title: "Jupyter中matplotlib中文乱码问题"
author: "wuyangzz"
tags: ["matplotlib"]
categories: ["Python"]
date: 2021-03-26T17:01:43+08:00
---
# 下载字体
首先下载中文字体，链接如下。

[字体下载链接](https://www.fontpalace.com/font-details/SimHei/)
![20210326170248](https://raw.githubusercontent.com/wuyangzz/blog_image/main/20210326170248.png)
win中下载号以后可以直接安装字体
# 修改matplotlibrc文件
matplotlibrc文件的位置如下：
```
D:\Anaconda3\Lib\site-packages\matplotlib\mpl-data
```
（1）删掉 font.family前面的 # 并改为SimHei
![20210326170504](https://raw.githubusercontent.com/wuyangzz/blog_image/main/20210326170504.png)


（2）删掉 font.sans-serif 前面的 #

    需要将刚才下载的SimHei添加到其中，如红色方框所示。
![20210326170531](https://raw.githubusercontent.com/wuyangzz/blog_image/main/20210326170531.png)


  保存退出！
# 把这个路径下的文件都删除（tex.cache和fontlist-.v310josn）

```c://用户//用户名//.matplotlib```
# 重新启动jupyter 