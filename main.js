const tf = require("@tensorflow/tfjs");

async function main(){
    var model = tf.sequential();
    model.add(tf.layers.dense({
        units : 1, // 输出空间的维度
        inputShape : [2]
    }))

    model.compile({
        loss : "meanSquaredError",
        optimizer : "sgd"
    })

    // 数据集准备
    let dataset = {
        x : [],
        y : [],
        z : [],
        count : 10
    }

    for(var i=0;i<dataset.count;i++) {
        let x = Math.floor(Math.random()*10);
        // 训练集打标签
        let y = Math.floor(Math.random()*10);
        let z = 2 * x + y - 1;
        dataset.x.push(x);
        dataset.y.push(y);
        dataset.z.push(z);
    }

    let inputData = [];
    for(var i=0;i<dataset.count;i++) {
        inputData.push([dataset.x[i], dataset.y[i]]);
    }

    let is = tf.tensor2d(inputData, [dataset.count, 2]);
    // let ys = tf.tensor2d(dataset.y, [dataset.count, 1]);
    let zs = tf.tensor2d(dataset.z, [dataset.count, 1]);

    let history = await model.fit(is, zs, {
        epochs : 500, // 训练次数
    })

    model.predict(tf.tensor2d([5, 6], [1,2])).print()
}

main();