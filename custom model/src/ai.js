import {TRAINING_DATA} from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/real-estate-data.js'

const INPUTS = TRAINING_DATA.inputs;
const OUTPUTS = TRAINING_DATA.outputs;

tf.util.shuffleCombo(INPUTS,OUTPUTS);

// Tensors
const INPUTS_TENSOR = tf.tensor2d(INPUTS)
const OUTPUTS_TENSOR = tf.tensor1d(OUTPUTS)

// Normalize
function normalize(tensor){
    const result = tf.tidy(function(){
        const MIN_VALUES = tf.min(tensor,0)
        const MAX_VALUES = tf.max(tensor,0)
        const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor,MIN_VALUES)
        const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);

        const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);

        return {NORMALIZED_VALUES, MIN_VALUES, MAX_VALUES}
    });
    return result;
}

const FEATURE_RESULTS = await normalize(INPUTS_TENSOR);
console.log('Normalized Values: ')
FEATURE_RESULTS.NORMALIZED_VALUES.print()
console.log('Min Values: ')
FEATURE_RESULTS.MIN_VALUES.print()
console.log('Max Values: ')
FEATURE_RESULTS.MAX_VALUES.print()

INPUTS_TENSOR.dispose()

// Model Architecture

const model = tf.sequential();
model.add(tf.layers.dense({inputShape: [2], units: 1, activation: 'relu'}));

model.summary()

// Train
const LEARNING_RATE = 0.1;
model.compile({
    optimizer: tf.train.sgd(LEARNING_RATE),
    loss: 'meanSquaredError'
});

let results = await model.fit(FEATURE_RESULTS.NORMALIZED_VALUES,OUTPUTS_TENSOR, {
    validationSplit: 0.151,
    shuffle: true,
    batchSize: 64,
    epochs: 15
})
console.log("Average error loss: " + Math.sqrt(results.history.loss[results.history.loss.length - 1]))
tf.tidy(()=>{
    let newInput = normalize(tf.tensor2d([[800, 2]]))

    let output = model.predict(newInput.NORMALIZED_VALUES)
    output.print();
});

export {normalize}

