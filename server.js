const path = require('path');
const url = require('url');
const express = require('express')
const multer = require('multer')
const jpeg = require('jpeg-js')
const PNG = require('pngjs').PNG;
const tf = require('@tensorflow/tfjs-node')
const nsfw = require('nsfwjs')

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

app.post('/nsfw', upload.single("image"), async (req, res) => {
  if (!req.file)
    res.status(400).send("Missing image multipart/form-data")
  else {
    try {
      const fileType = req.file.mimetype;
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
  let modelPath = url.pathToFileURL(path.resolve(__dirname, './model/web_model/')).pathname;
  modelPath = `file:/${modelPath}/`
  _model = await nsfw.load(modelPath, { type: 'graph' })
}

// Keep the model in memory, make sure it's loaded only once
load_model().then(() => app.listen(8080))
