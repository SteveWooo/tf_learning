const tf = require("@tensorflow/tfjs");
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
    label : fs.readFileSync(`${__dirname}/../dataset/mnist/mnist_labels_uint8`)
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
            gm(`${__dirname}/../dataset/mnist/mnist_images.png`)
            .crop(28*28, 1, 0, i)
            .write(`${__dirname}/dataset/${i}.png`, function(err) {
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

async function main(){
    var model = tf.sequential();
    
    // await handleData();

    console.log(dataset.label.length)

    // app.get('/test.png', function(req, res){
    //     res.send(dataset.image);
    // })

    // app.listen(81);
}

main();