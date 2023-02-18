import path, { dirname } from 'path';
import os from 'os'
import url, { fileURLToPath } from 'url'
import express from 'express'
import multer from 'multer'
import jpeg from 'jpeg-js';
import Sharp from 'sharp';
import tf from '@tensorflow/tfjs-node'
import nsfw from 'nsfwjs';
import { fileTypeFromBuffer } from 'file-type'

const app = express()
const upload = multer()

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
  return new Promise((resolve, reject) => {
    Sharp(image)
      .modulate({
        saturation: 0.7
      })
      .blur(2)
      .resize({width: 512})
      .toFormat('jpeg')
      .toBuffer({resolveWithObject: true})
      .then(({data, info}) => {
        const imageData = jpeg.decode(data, true);
        resolve(imageData);
      })
      .catch(err => { reject(err) });
  });
}

const SUPPORTED_TYPES = ['image/png', 'image/jpeg', 'image/webp'];

app.post('/nsfw', upload.single("image"), async (req, res) => {
  if (!req.file)
    res.status(400).send("Missing image multipart/form-data")
  else {
    try {
      let fileType = req.file.mimetype;

      if (!SUPPORTED_TYPES.includes(fileType.toLowerCase())) {
        res.json({
          code: -1,
          message: 'Not supported file type',
        })
        return;
      }

      try {
        const { ext, mime } = await fileTypeFromBuffer(req.file.buffer);
        fileType = mime;

        if (!SUPPORTED_TYPES.includes(fileType.toLowerCase())) {
          res.json({
            code: -1,
            message: 'Mismatch file type',
          })
          return;
        }
      } catch (error) {
        console.error(error);
      }


      const imageData = await decodeImage(fileType, req.file.buffer);
      const image = await convert(imageData)
      const predictions = await _model.classify(image)
      image.dispose()
      res.json(predictions);
    } catch (error) {
      console.error(error);
      res.json({
        code: -1,
        message: error.message
      })
    }

  }
});

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

// Keep the model in memory, make sure it's loaded only once
load_model().then(() => app.listen(8080))
