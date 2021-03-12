---
title: "Ubuntu新增用户和一键ssh免密登录脚本"
author: "wuyangzz"
tags: ["免密登录","ssh"]
categories: ["ubuntu"]
date: 2021-03-12T17:03:57+08:00
---
# 一键创建用户脚本
此脚本用户一键创建具有sudo权限的账户 并且可以免密使用sudo
```shell
id openpai &>/dev/null   #验证用户是否存在
if [ $? -eq 0 ];then
		echo "用户已经创建了"
	else
		echo "用户即将开始创建"
		useradd  -m -s /bin/bash openpai
		echo "用户创建成功"
		echo "正在设置密码"
		echo "openpai:openpai" | sudo chpasswd #给用户设置密码
		echo "密码设置成功"
		echo "openpai"

fi
echo "正在设置sudo权限"
mkdir /etc/sudoers.d/
echo openpai | sudo -S sh -c "echo 'openpai ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers.d/user-openpai"
sudo chmod 440 /etc/sudoers.d/user-openpai
```

# 批量ssh免密登录脚本
```shell
#!/usr/bin/bash

shell_dir=$(dirname "$0")
SHELL_DIR_ABSOLUTE=$(cd "$shell_dir"; pwd)
EXIST_DOWN_HOST="NO"
SSH_HOME_DIR="/root"

if ! which expect &>/dev/null; then
  if ! rpm -q tcl &>/dev/null; then
    [ -e "$SHELL_DIR_ABSOLUTE"/tcl-8.5.13-8.el7.x86_64.rpm ] && rpm -Uvh "$SHELL_DIR_ABSOLUTE"/tcl-8.5.13-8.el7.x86_64.rpm
  fi
  if ! rpm -q expect &>/dev/null; then
    [ -e "$SHELL_DIR_ABSOLUTE"/expect-5.45-14.el7_1.x86_64.rpm ] && rpm -Uvh "$SHELL_DIR_ABSOLUTE"/expect-5.45-14.el7_1.x86_64.rpm
  fi
fi

if ! which expect &>/dev/null; then
  echo "No expect is installed on the system"
  exit 1
fi

if ! [ -e "$SHELL_DIR_ABSOLUTE"/host.list ]; then
  echo "file $SHELL_DIR_ABSOLUTE/host.list does not exist"
  exit 1
fi

if [ "$USER" != "root" ]; then
  SSH_HOME_DIR="/home/$USER"
fi

[ -e $SSH_HOME_DIR/.ssh/id_rsa.pub -a ! -e $SSH_HOME_DIR/.ssh/id_rsa ] && rm -rf $SSH_HOME_DIR/.ssh/id_rsa.pub
[ ! -e $SSH_HOME_DIR/.ssh/id_rsa.pub -a -e $SSH_HOME_DIR/.ssh/id_rsa ] && rm -rf $SSH_HOME_DIR/.ssh/id_rsa
[ -e $SSH_HOME_DIR/.ssh/id_rsa.pub -a -e $SSH_HOME_DIR/.ssh/id_rsa ] || ssh-keygen -t rsa -f $SSH_HOME_DIR/.ssh/id_rsa -N '' >/dev/null 2>&1

while read host_info
do
  host_ip=$(echo $host_info | cut -d ' ' -f 1)
  if ping -c 1 -w 3 $host_ip &>/dev/null; then
    echo "  up -> $host_ip"
  else
    echo -e "\033[31mdown -> $host_ip\033[0m"
    EXIST_DOWN_HOST="YES"
  fi
done < "$SHELL_DIR_ABSOLUTE"/host.list

unset host_info
unset host_ip

if [ "$EXIST_DOWN_HOST" != "NO" ]; then
  echo "There are servers that cannot be connected"
  exit 1
fi

while read host_info
do
host_ip=$(echo $host_info | cut -d ' ' -f 1)
host_username=$(echo $host_info | cut -d ' ' -f 2)
host_password=$(echo $host_info | cut -d ' ' -f 3)
expect <<-EXPECT
set timeout -1
spawn ssh-copy-id $host_username@$host_ip
expect {
  "*(yes/no)" {
    send "yes\r"
    exp_continue
  }
  "*password:" {
    send "$host_password\r"
    exp_continue
  }
  eof {
    exit 0
  }
}
EXPECT
done < "$SHELL_DIR_ABSOLUTE"/host.list
unset host_info
unset host_ip
unset host_username
unset host_password

while read host_info
do
{
host_ip=$(echo $host_info | cut -d ' ' -f 1)
host_username=$(echo $host_info | cut -d ' ' -f 2)
echo "正在测试 $host_username@$host_ip ..."
if timeout 4 ssh $host_username@$host_ip "echo 0 $>/dev/null"; then
  echo "$host_username@$host_ip 测试成功 √"
else
  echo -e "\n\n\033[31m$host_username@$host_ip cannot be ssh connected\033[0m\n\n"
  echo 1 > /tmp/8585324c6aa741ad8d3ebebb4aa9c7ba.check.ssh
fi
}&
done < "$SHELL_DIR_ABSOLUTE"/host.list
wait

if [ -e /tmp/8585324c6aa741ad8d3ebebb4aa9c7ba.check.ssh ]; then
  rm -rf /tmp/8585324c6aa741ad8d3ebebb4aa9c7ba.check.ssh
  echo -e "结果:测试失败 ×\n\n"
  exit 1
else
  echo -e "\n\n全部:测试成功 √\n\n"
fi

```
脚本需要使用host.list文件进行设置需要连接的主机地址以及用户名，例如：

ip 用户名 密码
```txt
192.168.1.2 openpai openpai
192.168.1.3 openpai openpai
192.168.1.99 openpai openpai
192.168.1.98 openpai openpai

```