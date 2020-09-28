const tf = require("@tensorflow/tfjs");
require('@tensorflow/tfjs-node');
const Canvas = require('canvas');
const fs = require('fs');

const IMG_W = 28;
const IMG_H = 28;
const NUM_CLASS = 10;
const NUM_TRAINING_DATA = 55000;
const NUM_TEST_DATA = 10000;

var dataset = {
    trainingData : new Float32Array(NUM_TRAINING_DATA * IMG_H * IMG_W),
    trainingLabel : new Float32Array(NUM_TRAINING_DATA * NUM_CLASS),
    testData : new Float32Array(NUM_TEST_DATA * IMG_H * IMG_W),
    testLabel : new Float32Array(NUM_TEST_DATA * NUM_CLASS)
}


/**
 * 把训练数据集，测试数据集加载到内存中的操作。
 */
async function loadData(){
    const conf = {
        /**
         * 由于Canvas每次只能读一定大小的文件，所以训练数据集的这张大图需要切割开来。
         * 这里是训练数据集的文件路径以及每个文件中包含的数据集。
         */
        trainingData : [{
            filepath : `${__dirname}/data/mnist/mnist_training_data1.png`,
            count : 30000
        }, {
            filepath : `${__dirname}/data/mnist/mnist_training_data2.png`,
            count : 25000
        }],

        /**
         * 测试数据集比较小，所以可以一次过读取出来
         */
        testData : [{
            filepath : `${__dirname}/data/mnist/mnist_test_data.png`,
            count : 10000
        }],
        
        /**
         * 这里的标签包含训练和测试数据集。
         */
        label : [{
            filepath : `${__dirname}/data/mnist/mnist_labels_uint8`,
        }]
    }

    /**
     * 先处理训练数据集
     */
    let trainCanvas = [];
    let trainCtx = [];

    // 先把数据读入Canvas
    for(var i=0;i<conf.trainingData.length;i++) {
        let canvas = Canvas.createCanvas(IMG_W * IMG_H, conf.trainingData[i].count);
        let ctx = canvas.getContext('2d');

        trainCanvas.push(canvas);
        trainCtx.push(ctx);

        let img = await Canvas.loadImage(conf.trainingData[i].filepath);
        ctx.drawImage(img, 0, 0, IMG_W * IMG_H, conf.trainingData[i].count);
    }

    // 然后把数据从Canvas中读出来
    for (var i=0;i<conf.trainingData.length;i++) {
        let data = trainCtx[i].getImageData(0, 0, IMG_W * IMG_H, conf.trainingData[i].count);
        for(var k=0; k<data.length/4; k+=4) {
            dataset.trainingData[k / 4] = data[k];
        }
    }

    /**
     * 然后用同样的方式处理测试数据集
     */
    let testCanvas = [];
    let testCtx = [];

    // 先把数据读入Canvas
    for(var i=0;i<conf.testData.length;i++) {
        let canvas = Canvas.createCanvas(IMG_W * IMG_H, conf.testData[i].count);
        let ctx = canvas.getContext('2d');

        testCanvas.push(canvas);
        testCtx.push(ctx);

        let img = await Canvas.loadImage(conf.testData[i].filepath);
        ctx.drawImage(img, 0, 0, IMG_W * IMG_H, conf.testData[i].count);
    }

    // 然后把数据从Canvas中读出来
    for (var i=0;i<conf.testData.length;i++) {
        let data = testCtx[i].getImageData(0, 0, IMG_W * IMG_H, conf.testData[i].count);
        // ⭐只需要拿红色通道出来即可，因为图片只有黑白。节省空间
        for(var k=0; k<data.length/4; k+=4) {
            dataset.testData[k / 4] = data[k];
        }
    }

    /**
     * 最后处理label
     * 标签集总共65万个，每10个代表一组数据。每组数据都10个二进制位。其中为1的下标代表该图片的值。
     */
    let labelData = fs.readFileSync(conf.label[0].filepath);
    dataset.trainingLabel = new Float32Array(labelData.slice(0, NUM_TRAINING_DATA * NUM_CLASS));
    dataset.testLabel = new Float32Array(labelData.slice(NUM_TRAINING_DATA * NUM_CLASS, NUM_TRAINING_DATA * NUM_CLASS + NUM_TEST_DATA * NUM_CLASS));
    
    return ;
}

async function training(){
    await loadData();
    console.log(`loaded data`)

    // 数据处理完，以下是模型搭建和训练
    var model = tf.sequential();
    model.add(tf.layers.flatten({
        inputShape: [IMG_H, IMG_W, 1]
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

    // 训练
    try {
        let trainingData = dataset.trainingData;
        let input = tf.tensor4d(trainingData,
            [trainingData.length / (IMG_H * IMG_W), IMG_H, IMG_W, 1]); // 图片数量，图片数据行数（高），图片列数（宽），通道数（只取了红色通道，所以只有1）
    
        let labelData = dataset.trainingLabel;
        let output = tf.tensor2d(labelData,
            [labelData.length / NUM_CLASS, NUM_CLASS]); // 标签数量，每组标签数量
        
        await model.fit(input, output, {
            batchSize : 520, // 每次训练520组。这个数量越高，内存开销越大，训练速度越快。
            epochs : 112, // 训练次数
            verbose : 1, // 日志显示粒度
            validationSplit : 0.15, // 好像是用来实时检测准确度的样本采样率。
        })
 
        /**
         * 把模型保存起来。
         * 这里坑挺大的，完全就是套了python的壳，在windows下必须把@tensorflow/tfjs-node安装好才能用这个函数
         */
        await model.save(`file://./model/mnist`);
    }catch(e) {
        console.log(e);
    }
}

/**
 * 测试训练模型。
 */
async function test(){
    await loadData();
    console.log(`loaded data`)

    // 把模型从本地拿出来
    let model = await tf.loadLayersModel(`file://./model/mnist/model.json`);
    let testData = dataset.testData;
    let result = model.predict(tf.tensor4d(testData, 
        [testData.length / (IMG_W * IMG_H), IMG_H, IMG_W, 1])); // 图片数量，图片数据行数（高），图片列数（宽），通道数（只取了红色通道，所以只有1）

    // 把结果转换为普通数组，方便下面处理
    result = await result.array(); 

    // 对比准确率
    let totalGapPercentage = 0; // 代表总误差率。gap也可以理解为errorGap
    for(var i=0;i<result.length;i++) {
        result[i] = new Float32Array(result[i]);
        // 把当前的测试标签拿出来。
        let testLabel = dataset.testLabel.slice(i * NUM_CLASS, i * NUM_CLASS + NUM_CLASS);
        
        // 表示与测试标签的差别。
        let gap = 0;
        for(var k=0;k<NUM_CLASS;k++) {
            gap += Math.abs(result[i][k] - testLabel[k]); // 差距需要取绝对值。
        }
        let gapPercentage = gap / 10;
        totalGapPercentage += gapPercentage;
        
    }

    let errorRate = totalGapPercentage / result.length;
    console.log(`correct Rate : ${1 - errorRate}`);
}

// training();
test();