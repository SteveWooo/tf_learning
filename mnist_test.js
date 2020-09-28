const tf = require("@tensorflow/tfjs");
require('@tensorflow/tfjs-node');
const Canvas = require('canvas');
const Express = require('express');
const app = Express();
const gm = require('gm');
const fs = require('fs');
const dataset = {
    /**
     * 图片下载地址：https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png
     * 标签下载地址：https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8
     */
    image : fs.readFileSync(`${__dirname}/../dataset/mnist/mnist_images.png`),
    label : fs.readFileSync(`${__dirname}/../dataset/mnist/mnist_labels_uint8`),

    trainLabel : fs.readFileSync(`${__dirname}/../dataset/mnist/t10k-labels.idx1-ubyte`),

    // 切割好的图片
    images : {
        training : [],
        test : []
    },
    labels : {
        training : [],
        test : [],
    }
}
const IMAGE_H = 28;
const IMAGE_W = 28;
const IMAGE_SIZE = IMAGE_H * IMAGE_W;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;

const NUM_TRAIN_ELEMENTS = 55000;
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

async function handleData(){
    // PNG头部编码处理方式
    // let fileHeader = dataset.image.slice(0, 16); // png文件头部编码
    // let imgHeaderLength = parseInt('0x' + fileHeader.slice(8, 12).toString("hex")); // 图片头部编码长度
    // let imgInfo = {
    //     imgHeader : dataset.image.slice(16, 16 + imgHeaderLength),
    //     width : 0,
    //     height : 0,
    //     body : []
    // }
    // imgInfo.body = dataset.image.slice(16 + imgHeaderLength, dataset.image.length);
    // imgInfo.width = parseInt('0x' + imgInfo.imgHeader.slice(0, 4).toString('hex'));
    // imgInfo.height = parseInt('0x' + imgInfo.imgHeader.slice(4, 8).toString('hex'));
    // console.log(imgInfo.body.length);

    function cut(i){
        return new Promise(resolve=>{
            gm(`${__dirname}/../originDataset/mnist/mnist_images.png`)
            .crop(28*28, 1, 0, i)
            .write(`${__dirname}/originDataset/${i}.png`, function(err) {
                if (err) {
                    console.log(err);
                }
                resolve();
            })
        })
    }

    const STEP = 20;
    for(var i=0;i<NUM_DATASET_ELEMENTS;i+=STEP) {
    // for(var i=0;i<30;i+=10) {
        let promises = [];
        for(var k=0;k<STEP;k++) {
            promises.push(cut(i + k));
        }
        await Promise.all(promises);
        console.log(`${i + STEP} / ${NUM_DATASET_ELEMENTS}`);
    }
}

// 上面的函数把图片切成一条一条，这里把它变成28x28
async function buildPic(){
    async function build(imgNum){
        let img = await Canvas.loadImage(`${__dirname}/originDataset/${imgNum}.png`);

        // 创建28x28的输出图片
        var canvas = Canvas.createCanvas(28, 28);
        var ctx = canvas.getContext('2d');
        for(var i=0;i<28;i++) {
            ctx.drawImage(img, i*28, 0, 28, 1, 0, i, 28, 1);
        }
        var base64Img = canvas.toBuffer('image/png', { compressionLevel: 3, filters: canvas.PNG_FILTER_NONE });
        fs.writeFileSync(`${__dirname}/dataset/${imgNum}.png`, base64Img);
    }
    // console.log(dataset.trainLabel.slice(0, 10));
    for(var i=0;i<65000;i++) {
        await build(i);
        console.log(`done ${i} / 65000`);
        // console.log(dataset.label.slice(i * 10, i * 10 + 10))
    }
    
    return ;
}

// 把mnist数据集图片切割为训练集和测试集，为了能让Canvas载入。。
async function cutMnistInTo2(){
    function cutTraining1(){
        return new Promise(resolve=>{
            gm(`${__dirname}/../dataset/mnist/mnist_images.png`)
            .crop(28*28, 30000, 0, 0)
            .write(`${__dirname}/../dataset/mnist/mnist_training_data1.png`, function(err) {
                if (err) {
                    console.log(err);
                }
                resolve();
            })
        })
    }

    function cutTraining2(){
        return new Promise(resolve=>{
            gm(`${__dirname}/../dataset/mnist/mnist_images.png`)
            .crop(28*28, 25000, 0, 30000)
            .write(`${__dirname}/../dataset/mnist/mnist_training_data2.png`, function(err) {
                if (err) {
                    console.log(err);
                }
                resolve();
            })
        })
    }

    function cutTest(){
        return new Promise(resolve=>{
            gm(`${__dirname}/../dataset/mnist/mnist_images.png`)
            .crop(28*28, 10000, 0, 55000)
            .write(`${__dirname}/../dataset/mnist/mnist_test_data.png`, function(err) {
                if (err) {
                    console.log(err);
                }
                resolve();
            })
        })
    }

    await cutTraining1();
    await cutTraining2();
    await cutTest();
}

// 把切割好的图片全部读进来，同时把标签切割一下
async function loadData(){
    // 好好用数组切啊。。。。
    let trainingData1Canvas = Canvas.createCanvas(28 * 28, 30000);
    let trainingData1Ctx = trainingData1Canvas.getContext('2d');
    let trainingData1 = await Canvas.loadImage(`${__dirname}/../dataset/mnist/mnist_training_data1.png`);
    trainingData1Ctx.drawImage(trainingData1, 0, 0, 28 * 28, 30000);
    let trainingDataRGB1 = trainingData1Ctx.getImageData(0, 0, 28 * 28, 30000);
    // for(var i=0;i<trainingDataRGB1.data.length;i++) {
    //     dataset.images.training.push(trainingDataRGB1.data[i]);
    // }
    // dataset.images.training = Array.prototype.slice.call(trainingDataRGB1.data);
    dataset.images.training = new Float32Array(30000 * 28 * 28);
    // 只需要读红色chanel
    for(var i=0;i<trainingDataRGB1.data.length/4;i+=4) {
        dataset.images.training[i/4] = trainingDataRGB1.data[i];
    }

    // let trainingData2Canvas = Canvas.createCanvas(28 * 28, 25000);
    // let trainingData2Ctx = trainingData2Canvas.getContext('2d');
    // let trainingData2 = await Canvas.loadImage(`${__dirname}/../dataset/mnist/mnist_training_data2.png`);
    // trainingData2Ctx.drawImage(trainingData2, 0, 0, 28 * 28, 25000);
    // let trainingDataRGB2 = trainingData2Ctx.getImageData(0, 0, 28 * 28, 25000);
    // // for(var i=0;i<25000;i++) {
    // //     dataset.images.training.push(trainingDataRGB2.data.slice(i*4*28*28, i*4*28*28 + 4*28*28));
    // // }
    // // console.log(dataset.images.training.length)
    // trainingDataRGB2.data = Array.prototype.slice.call(trainingDataRGB2.data);
    // console.log('concating')
    // dataset.images.training = dataset.images.training.concat(trainingDataRGB2.data);
    console.log(`loaded images`);

    let testData1Canvas = Canvas.createCanvas(28 * 28, 10000);
    let testData1Ctx = testData1Canvas.getContext('2d');
    let testData1 = await Canvas.loadImage(`${__dirname}/../dataset/mnist/mnist_test_data.png`);
    testData1Ctx.drawImage(testData1, 0, 0, 28 * 28, 10000);
    let testDataRGB1 = testData1Ctx.getImageData(0, 0, 28 * 28, 10000);
    // for(var i=0;i<trainingDataRGB1.data.length;i++) {
    //     dataset.images.training.push(trainingDataRGB1.data[i]);
    // }
    // dataset.images.training = Array.prototype.slice.call(trainingDataRGB1.data);
    dataset.images.test = new Float32Array(10000 * 28 * 28);
    // 只需要读红色chanel
    for(var i=0;i<testDataRGB1.data.length/4;i+=4) {
        dataset.images.test[i/4] = testDataRGB1.data[i];
    }
    console.log('loaded test Images.');

    // dataset.labels.training = dataset.label.slice(0, 550000);
    dataset.labels.training = new Float32Array(dataset.label.slice(0, 300000));
    dataset.labels.test = new Float32Array(dataset.label.slice(550000, 650000));
    console.log(`loaded labels`);
    return ;
}

async function main(){
    var model = tf.sequential();
    
    // await handleData(); // 废置
    // await buildPic(); // 废置
    // await cutMnistInTo2();

    await loadData();

    model.add(tf.layers.flatten({
        inputShape: [IMAGE_H, IMAGE_W, 1]
    }));
    
    model.add(tf.layers.dense({
        units: 42, activation: 'relu'
    }));
    model.add(tf.layers.dense({
        units: 10, activation: 'softmax'
    }));

    model.compile({
        optimizer : 'rmsprop',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });

    try{
        let trainingData = dataset.images.training;
        
        let input = tf.tensor4d(trainingData,
            [trainingData.length / (28*28), 28, 28, 1]) // hight, width
        
        let labelData = dataset.labels.training;
        let output = tf.tensor2d(labelData,
            [labelData.length / NUM_CLASSES, NUM_CLASSES]);
        
        // console.log(output)

        await model.fit(input, output, {
            batchSize : 520,
            epochs : 112,
            verbose : 1,
            validationSplit : 0.15,
        })

        await model.save(`file://./model`);
    }catch(e) {
        console.log(e)
    }

}

async function test(){
    await loadData();
    let model = await tf.loadLayersModel(`file://./model/model.json`);
    // let testData = dataset.images.test.slice(0, 2*28*28);
    let testData = dataset.images.test;
    let res = model.predict(tf.tensor4d(testData,
        [testData.length / (28*28), 28, 28, 1]));
    res = await res.array();
    // console.log(res.length)
    // console.log(dataset.labels.test);

    let totalGapPercentage = 0;
    for(var i=0;i<res.length;i++) {
        res[i] = new Float32Array(res[i]);
        // console.log(res[i]);
        let testLabel = dataset.labels.test.slice(i * NUM_CLASSES, i * NUM_CLASSES + NUM_CLASSES);
        // console.log(testLabel)
        let gap = 0;
        for(var k=0;k<NUM_CLASSES;k++) {
            gap += Math.abs(res[i][k] - testLabel[k]);
        }
        let gapPercentage = gap / 10;
        totalGapPercentage += gapPercentage;
        // console.log(gapPercentage)
    }

    let errorRate = totalGapPercentage / res.length;
    console.log(`correct Rate : ${1 - errorRate}`);
}

// main();
test();