import {normalize} from './ai'
let model = await tf.loadLayersModel('/ai.json')
let newValue = tf.tensor2d([[800, 2]])
let result = normalize(newValue)
let output = await model.predict(result.NORMALIZED_VALUES)
output.print()