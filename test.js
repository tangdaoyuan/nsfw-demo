import path, { dirname } from 'path';
import os from 'os'
import url, { fileURLToPath } from 'url'
import jpeg from 'jpeg-js';
import Sharp from 'sharp';
import { PNG } from 'pngjs'
import tf from '@tensorflow/tfjs-node'
import nsfw from 'nsfwjs';
import { fileTypeFromBuffer } from 'file-type'
import fs from 'fs/promises'

let _model

const convert = async (image) => {
  const numChannels = 3
  const numPixels = image.width * image.height
  const values = new Int32Array(numPixels * numChannels)

  for (let i = 0; i < numPixels; i++)
    for (let c = 0; c < numChannels; ++c)
      values[i * numChannels + c] = image.data[i * 4 + c]

  return tf.tensor3d(values, [image.height, image.width, numChannels], 'int32')
}


const decodeImage = function (imageType, image) {
    if (imageType === 'image/webp') {
       return new Promise((resolve, reject) => {
          Sharp(image)
            .resize({width: 512})
            .toFormat('jpeg')
            .toBuffer({resolveWithObject: true})
            .then(({data, info}) => {
              const imageData = jpeg.decode(data, true);
              resolve(imageData)
            })
            .catch(err => { reject(err) });
       });
    }

  if (imageType === 'image/png') {
    return new Promise((resolve, reject) => {
      new PNG({ filterType: 4 }).parse(image, async function (error, data) {
        if (!error) {
          resolve(data);
        } else {
          reject(new Error('invalid image/png format'));
        }
      });
    })
  }

  return jpeg.decode(image, true);
}

const SUPPORTED_TYPES = ['image/png', 'image/jpeg', 'image/webp'];


async function testSingleFile(buf, cb) {
  let fileType = '';
  try {
    try {
      const { ext, mime } = await fileTypeFromBuffer(buf);
      fileType = mime;
  
      if (!SUPPORTED_TYPES.includes(fileType.toLowerCase())) {
        console.err('Mismatch file type')
        return;
      }
    } catch (error) {
      console.error(error);
    }
  
  
    const imageData = await decodeImage(fileType, buf);
    const image = await convert(imageData)
    const predictions = await _model.classify(image)
    image.dispose()

    cb(predictions)
  } catch (error) {
    console.error(error);
  }
}


const dataDir = './Data';

async function test(verify) {
  const absDir = path.resolve(dataDir);
  const dirs = await fs.readdir(absDir);
  dirs.forEach(async (dir) => {
     const absImagePath = path.resolve(absDir, dir);
     const buf = await fs.readFile(absImagePath)
     await testSingleFile(buf, verify)
  });
}



const load_model = async () => {
  const __dirname = fileURLToPath(dirname(import.meta.url));
  let modelPath = url.pathToFileURL(path.resolve(__dirname, './model/web_model/')).pathname;
  if (os.platform() === 'win32') {
    modelPath = `file:/${modelPath}/`
  } else {
    modelPath = `file://${modelPath}/`
  }

  _model = await nsfw.load(modelPath, { type: 'graph' })
}

load_model().then(() => {
  test(prediction => {
    console.log(prediction)
  })
})
