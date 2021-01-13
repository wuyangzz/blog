---
title: "实验室GPU服务器操作细则"
author: "wuyangzz"
categories: ["实验室GPU"]
tags: ["GPU"]
date: 2021-01-10T10:42:55+08:00
---
# 实验室GPU服务器操作细则



# 实验室GPU简介

- 系统 ubuntu18.04
- IP:172.23.253.* 172.23.253.15* 
- 双路3090显卡

# docker

### docker 简介

​	docker镜像可以看作是一个以及配置好了很多环境的操作系统，docker与虚拟机类似，但是两者在原理上有很大的不同。docker是讲操作系统的底层虚拟化，而虚拟机是将硬件虚拟化，因此docker具有更高的便携性和跟高效的利用服务器的性能。同时由于docker的标准化，它可以无视任何基础设施的标志，可以很简单的部署到任何的一个地方，另外docker重要的优点就是可以提供良好的隔离兼容。

​	其主要概念中最重要的就是为**images** **container** 

- ###### Images 是一个只读的模版，可以用来创建container，可以直接下载已经构建好的image，也可以自己通过Dockerfile来创建。

- ###### container 是image的可运行实例，其可以通过API和CLI(命令行)进行操作。

### NGC

​	NGC是NVIDIA官方提供的容器，其主要的作用是为用户提供一个简单、高效、安全的镜像，方便用户可以最轻松的使用NVIDIA GPU。



[使用Docker CLI从NGC容器注册表中提取容器](https://docs.nvidia.com/ngc/ngc-catalog-user-guide/index.html#pull_ngc_docker)

1. 打开[NGC网站](https://ngc.nvidia.com/catalog/containers?orderBy=modifiedDESC&pageNumber=0&query=&quickFilter=containers&filters=)，其中可以浏览自己所需要的容器

![image-20201206195516130](https://raw.githubusercontent.com/wycyz1/blog_image/main/20201206195516.png)

2. 自己根据自己的基础环境 如Tensorflow Pytorch进行搜索。并进入其中。里面有该容器的参考文档。只需要将其Pull记住。

![image-20201207181045933](https://raw.githubusercontent.com/wycyz1/blog_image/main/20201207181251.png)

3. 我们一般会提供一个使用jupyter的8888端口一个使用xshell和xftp的22端口。并且将用户名和密码告知。

   

### docker命令

   `dockers ps -a` **查看容器**

   `docker images` 查看镜像

   `docker start 容器id`   **启动容器**

   `docker attach 容器id`  **进入容器**

  `docker stop 容器id`  **停止容器**

   `docker rm 容器id`  **删除容器**

  ` docker image rm`  **镜像id  删除镜像id**

   `docker load` 本地镜像  **导入本地镜像**

   ##### ubuntu 基本命令

  ` ps -ef |grep 程序名 ` **查看正在运行的程序**

   `kill -9 程序pid`  **杀死改pid的程序**

   `nohup jupyter-lab --ip 0.0.0.0 --port 8888 --allow-root > jupyter.log 2>&1 &` **运行jupyter**

  ` sudo apt-get install openssh-server` **安装ssh服务器**

   **配置ssh客户端**，去掉PasswordAuthentication yes前面的#号，保存退出

   `sudo vi /etc/ssh/ssh_config`

   ​	把PermitRootLogin prohibit-password改成PermitRootLogin yes

   **重启ssh服务**

   ​	`sudo /etc/init.d/ssh restart`

  ` passwd root` **修改root密码**

   1. *## step 1：终端输入*
   2. jupyter notebook *--generate -config*
   3. *## step 2：终端输入*
   4. jupyter notebook password    *# 按提示，输入密码*



```
docker run --gpus "device=0" --name=202022000327 -it -v  "/docker/contriner_dir/202022000327/":"/workspace" -p 32722:22 -p 32788:8888 6c3
```



docker中 启动所有的容器命令

```javascript
docker start $(docker ps -a | awk '{ print $1}' | tail -n +2)
```

docker中    关闭所有的容器命令

```javascript
docker stop $(docker ps -a | awk '{ print $1}' | tail -n +2)
```

docker中 删除所有的容器命令

```javascript
docker rm $(docker ps -a | awk '{ print $1}' | tail -n +2)
```

docker中    删除所有的镜像

```javascript
docker rmi $(docker images | awk '{print $3}' |tail -n +2)
```

1. 修改root账户密码 

	``` passwd root ```

2. 更新sudo权限  

	``` apt-get update apt-get install sudo ```

	

3. 更新软件  sudo apt-get update

4. 更新依赖 sudo apt-get upgrade

5. 安装ssh服务 sudo apt-get install openssh-server

6. vim /etc/ssh/sshd_config  

   #PermitRootLogin prohibit-password

   PermitRootLogin yes

7. 重启ssh服务   /etc/init.d/ssh restart 

8. 自启 ssh sudo systemctl enable ssh  

9. 查看ssh和从host端进入的是否环境变量一致

   echo $PATH

   如果不一样就

   vim /etc/profile

   在后面加入host段的环境变量  

   export PATH="$PATH:**/usr/local/**"

   然后哎 source /etc/profile 或者重启

