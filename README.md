# Mnist数据集在Nodejs下使用TF框架DEMO
在windows下使用nodejs完成mnist数据集训练工程的记录。工程内涵括数据集
### 安装依赖
该工程在windwos下可完美运作，前提是需要做好充分的安装准备。
#### 1. 首先安装 windows-build-tools
为了安装tensorflow，需要在windows安装以下依赖，因为Nodejs下的TF大部分还是依赖原生版本的TF，需要编译。
```
npm install --global --production windows-build-tools
```
#### 2. 安装TF依赖
核心框架安装
```
npm i --save @tensorflow/tfjs
npm i --save @tensorflow/tfjs-node
```
#### 3. 安装Canvas
Nodejs对图片的处理能力偏弱，但Canvas还是蛮好用的，本工程选择使用服务端侧的画布来操作图片。（Canvas有个坑，就是读图片有大小限制）
```
npm i --save canvas
```

### 开始
###### 入口文件为mnist.js，里面包含两个函数，分别包含训练和测试。
```
traing();
test(); 
```

###### 首先是训练training()，它需要调用训练数据集，而数据集保存在以下目录。
```
./data/mnist
```
数据集里面包括了：
1. 65000组标签集合（mnist_labels_uint8），其中55000为训练集的标签，10000为测试集标签。每组标签由10个二进制组成，每组标签只有一个1，这个1的下标就代表这个手写图片的数字（0到9）。
2. 10000条测试数据集（mnist_test_data.png）
3. 30000 + 25000 = 55000 条训练数据集(mnist_training_data1.png, mnist_training_data2.png)。因为Canvas读不下全部数据（全部也就共10M+，Canvas读8M就歇菜了），所以这里我提前使用gm把图片切割了一下，大家尽管用就可以了。

###### 然后自己构建模型，进行训练
代码里的model.fit函数

###### 训练完成后，模型会保存在
```
await model.save(`file://./model/mnist`);
```
这里坑挺大的，连文件保存都是直接给原生代码套壳的感觉。

###### 测试的时候，会把模型调出来，同时把测试数据集导入进去
```
let model = await tf.loadLayersModel(`file://./model/mnist/model.json`);
```
然后test就会把这个模型调出来，然后跑predict函数，再对比。

### 其他
main.js 是一个二元一次方程的入门demo，适合我这种憨憨学，大神们大可不必吐槽。