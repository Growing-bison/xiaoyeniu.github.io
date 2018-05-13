---
title: "关于python自定义包"
date: 2018-02-09T16:16:59+08:00
draft: false
tags: ["python","自定义包","模式匹配"]
share: true
---

这是关于如何建立python的自定义包的简单笔记，供简单参考。
<!--more-->


## 关于python自定义包

- **使用的环境说明**
```{python}
# !/usr/bin/env python3.5
# operation system: windows7
```


#### 一、模块基本结构与引用
　　常见的一个模块结构如下，一般还需要有一个test.py文件与package文件同目录，来测试这个包。
![Alt text](/img/jietu00.jpg)


另外，`__int__.py`文件是必须要有，关键！它的存在表示此目录应该被作为一个package（包）。
在需要引用时，进入包的顶层目录rstools，导入某个函数。
package-rstools
module-imagestretch
方法/函数-get_no_data
```{python}
from rstools.imagestretch import get_no_data
get_no_data(file) 
# 当以*导入，package内的module受__int__.py限制。
``` 
#### 二、package内部互相调用
```{python}
package1/
       __init__.py
       subPack1/
               __init__.py
               module_11.py
               module_12.py
               module_13.py
      subPack2/
               __init__.py
               module_21.py
               module_22.py
```
- **1**、调用同一个package中的module，可直接import。如module_12.py希望调用module_11
```{python}
import module_11
```
- **2**、调用非同一个package中的module，如module_11.py希望导入module_21.py中的FuncA
```{python}
from subPack2.module_21 import module_11
```

#### 三、\__int__.py文件
该文件决定了包的导入和使用方式，需要很好设计。

- **1**: __all__属性
```{python}
__int__=['dataprepare','datapansharpen']
```
- **2**: 

#### 四、setput.py文件
**文件中几个主要参数**
- name：str，包名称
- version：str，版本号码
- url：包的链接，通常为 Github 上的链接，或者是 readthedocs 链接
- packages：需要包含的子包列表，find_packages()助查找
- setup_requires：指定依赖项
- test_suite：测试时运行的工具
- long_description：在一个来源处做说明介绍
```{python}
long_description=open('README.md').read()
```

- **1**：打包某个模块
```{python}
# 打包项目中某一个模块
from distutils.core import setup
from setuptools import find_packages
setup(
name='production',
version='v1.0',
packages=find_packages(exclude=['production.moving_windows'])
)
```
- **2**：打包整个项目
```{python}
from setuptools import setup, find_packages 
setup(name = 'production',version = '1.0',
      py_modules = ['production.production_whole'],
      author = 'lqkpython',
      author_email = 'luqikun@gagogroup.com',
      url = 'http://www.gagogroup.com',
      description = 'A simple calculation for grass production ',
      packages=find_packages(exclude=['*.*']))
```

#### 五、将项目进行打包
　　整个项目是production/module，现在希望对其进行打包，从而能够为他人分享安装.

- **1**、准备好的文件

![Alt text](/img/1512714688155.png)
![代码所在](/img/1512714788732.png)

- **2**、打包整个项目

```flow
st=>start: 安装好打包所需的模块，setuptools模块
op1=>operation: 在module下的setup.py文件准备
op2=>operation: 编译:'python setup.py build'
io1=>inputoutput: 在setup.py相同目录下多出build目录
op3=>operation: 打包:'python setup.py sdist'
io2=>inputoutput: 打包后会在setup.py同目录下多出一个disk目录，存放打好的包
io3=>inputoutput: 完成打包
op4=>operation: 部署安装:'python setup.py install'
e=>end: 完成整个项目打包

st->op1->op2->io1->op3->io2->op4->e

```

------
补充的内容：
1. 将代码组织成包,想用import语句从另一个包名没有硬编码过的包的中导入子模块。
https://www.kancloud.cn/kancloud/python3-cookbook/47306



------
练习文件链接如下
[modul](/files/module.zip)

------

###### 参考文档链接
- **1**: python中自定义包的导入和使用  http://blog.csdn.net/maryhuan/article/details/22048763
- **2**: Python包的编写和使用 http://blog.csdn.net/destinyuan/article/details/51704935
- **3**: python 创建自己的包 http://blog.csdn.net/dai_jing/article/details/46005729
- **4**:  python 打包与部署 http://www.cnblogs.com/perfei/p/5139543.html
- **5**:  [这篇文章教会你打包Python程序](http://mp.weixin.qq.com/s?src=3&timestamp=1512699606&ver=1&signature=2qvUQsQ6Tzf13kTij4VZ4cULEA7t1XgK8B6Ny*FKurCh00Ks7vfJDeaVlf19Zraxla6fewFCf1twe*MG2MHyuHJ923DZsNUmK-5ZxAp*NGtPKXN-dd7apitMEn7CSN2fcEDOip6ZEyi4Ku0hLXpnmAz7YSG4zWZKUyE6DiHxhfA=)
- **6**: [Python编写和发布一个Python CLI工具](http://www.jianshu.com/p/085e062e4db0)