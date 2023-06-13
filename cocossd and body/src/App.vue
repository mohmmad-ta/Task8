<script setup>
import {ref} from "vue";
const image = ref()
const points = ref()
const hwImg = ref()
const yxImg = ref()
import {modelCoco, modelBody} from "@/ai";

const run = async ()=>{
  let imageTensor = await tf.browser.fromPixels(image.value)
  let cropStartPoint = [0,0,0]
  let cropSize = [0,0,3]
  await modelCoco.detect(imageTensor).then(predictions => {
    for (let i=0; i<predictions.length;i++){
      if (predictions[i].class === "person"){
        cropStartPoint[0] = Math.floor(predictions[i].bbox[1])
        cropStartPoint[1] = Math.floor(predictions[i].bbox[0])
        cropSize[0] = Math.floor(predictions[i].bbox[3])
        cropSize[1] = Math.floor(predictions[i].bbox[2])
      }
    }
  });
  let croppedTensor = tf.slice(imageTensor, cropStartPoint, cropSize)
  let resizedTensor = tf.image.resizeBilinear(croppedTensor,[192,192], true).toInt()
  let tensorOutput = await modelBody.predict(tf.expandDims(resizedTensor));
  let arrayOutput = await tensorOutput.array()
  points.value=arrayOutput[0][0]
  yxImg.value= cropStartPoint
  hwImg.value= cropSize
  console.log(points.value[0])
}
</script>

<template>
  <div class="w-full h-screen bg-gray-800 flex justify-center items-center">
    <div class="flex flex-col justify-center relative">
      <img class="relative" crossorigin="anonymous" ref="image" src="https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/standing.jpg" alt="">
      <div
          v-for="point in points" :key="points"
          class="bg-cyan-500 w-2.5 h-2.5 rounded-full absolute"
          :style="{top: point[0]*hwImg[0]+yxImg[0]-5 +'px',left: point[1]*hwImg[1]+yxImg[1]-5 +'px'}"
      ></div>
      <button @click="run" type="button" class="text-white bg-violet-500 p-2 rounded-sm mt-5">Run Model</button>
    </div>
  </div>
</template>
