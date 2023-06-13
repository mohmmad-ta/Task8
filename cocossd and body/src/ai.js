const MODEL_PATH = 'https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4'
const modelCoco = await cocoSsd.load()
const modelBody = await tf.loadGraphModel(MODEL_PATH, { fromTFHub: true })
export {modelCoco, modelBody}