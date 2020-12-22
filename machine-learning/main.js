const tf = require('@tensorflow/tfjs-node');

// dataset
const data = require('./winequality.js');

//model, 4 dense layers
function createModel() {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [11],
      units: 50,
      useBias: true,
      activation: 'relu',
    }),
  );

  model.add(tf.layers.dense({ units: 30, useBias: true, activation: 'tanh' }));
  model.add(tf.layers.dense({ units: 20, useBias: true, activation: 'relu' }));
  model.add(
    tf.layers.dense({ units: 10, useBias: true, activation: 'softmax' }),
  );


  return model;
}

//data formatting
function extractInputs(data) {
  let inputs = [];
  inputs = data.map((d) => [
    d.fixed_acidity,
    d.volatile_acidity,
    d.citric_acid,
    d.residual_sugar,
    d.chlorides,
    d.free_sulfur_dioxide,
    d.total_sulfur_dioxide,
    d.density,
    d.pH,
    d.sulphates,
    d.alcohol,
  ]);
  return inputs;
}

//tensorflow tensor creation
function prepareData(data) {
  return tf.tidy(() => {
    tf.util.shuffle(data);

    const inputs = extractInputs(data);
    const outputs = data.map((d) => d.quality);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, inputs[0].length]);
    const outputTensor = tf.oneHot(tf.tensor1d(outputs, 'int32'), 10);

    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const outputMax = outputTensor.max();
    const outputMin = outputTensor.min();

    const normalizedInputs = inputTensor
      .sub(inputMin)
      .div(inputMax.sub(inputMin));
    const normalizedoutputs = outputTensor
      .sub(outputMin)
      .div(outputMax.sub(outputMin));

    return {
      inputs: normalizedInputs,
      outputs: normalizedoutputs,
      inputMax,
      inputMin,
      outputMax,
      outputMin,
    };
  });
}

async function trainModel(model, inputs, outputs, epochs) {
  model.compile({
    optimizer: tf.train.adam(),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  const batchSize = 64;

  return await model.fit(inputs, outputs, {
    batchSize,
    epochs,
    shuffle: true,
  });
}

async function evaluateModel(model, inputs, outputs) {
  const result = await model.evaluate(inputs, outputs, { batchSize: 64 });
  console.log('Accuracy:');
  result[1].print();
}

async function run() {
  const model = createModel();

  const tensorData = prepareData(data);
  const { inputs, outputs } = tensorData;

  await trainModel(model, inputs, outputs, 100);
  console.log('Finished Training');

  await evaluateModel(model, inputs, outputs);
}

run();
