import path, { dirname } from 'path';
import os from 'os'
import url, { fileURLToPath } from 'url'
import jpeg from 'jpeg-js';
import Sharp from 'sharp';
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
    return new Promise((resolve, reject) => {
      Sharp(image)
        .modulate({
          saturation: 0.7,
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
  
    const imageData1 = await decodeImage(fileType, buf);
    const image1 = await convert(imageData1)
    const predictions1 = await _model.classify(image1)
    image1.dispose()

    cb(predictions1)
  } catch (error) {
    console.error(error);
  }
}


async function test(dataDir, verify) {
  const absDir = path.resolve(dataDir);
  const dirs = await fs.readdir(absDir);

  for (let dir of dirs) {
    const absImagePath = path.resolve(absDir, dir);
    const buf = await fs.readFile(absImagePath)
    await testSingleFile(buf, (prediction1, prediction2) => {
      verify(absImagePath, prediction1, prediction2)
    })
  }
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

const negDir = './Data/negative';
const rawDir = './Data/raw/pictrues';
const sexyDir = './Data/sexy';


// the smaller the Sexy value, the more SFW
function doubleCheckSFW(predictions) {
  return predictions.some(pred => {
    const {className, probability} = pred;
    if (className === 'Sexy') {
      // sfw
      return probability > 0.01;
    }

    return false;
  })
}


const PORN_THRESHOLD = 0.7

async function sexyTest(copyFile = false) {
  let sexyRets = [];
  await test(sexyDir, async (filePath, predictions) => {
    const ret = predictions.some(pred => {
      const {className, probability} = pred;
      return ['Porn'].includes(className) && probability > PORN_THRESHOLD;
    });

    if (ret) {
      sexyRets.push({
        filePath,
        predictions: predictions.reduce((acc, cur) => {
            acc[cur.className] = cur.probability;
            return acc;
          }, {})
      });
    }
  });
  await fs.writeFile('./sexy-flaw.json', JSON.stringify(sexyRets, null, 2));

  if (copyFile) {
    for (let ret of sexyRets) {
      await fs.copyFile(ret.filePath, path.resolve('./temp', path.basename(ret.filePath)));
    }
  }
}

async function normalTest() {
  let normalRets = [];
  await test(rawDir, async (filePath, predictions) => {
    const ret = predictions.some(pred => {
      const {className, probability} = pred;
      return ['Porn'].includes(className) && probability > PORN_THRESHOLD;
    });
    if (ret) {
      normalRets.push({
        filePath,
        predictions: predictions.reduce((acc, cur) => {
            acc[cur.className] = cur.probability;
            return acc;
          }, {})
      });
    }
  });
  await fs.writeFile('./normal-flaw.json', JSON.stringify(normalRets, null, 2));
}


async function negTest(copyFile = false) {
  const negRets = [];
  await test(negDir, (filePath, predictions) => {
    const ret = predictions.every(pred => {
      const {className, probability} = pred;
      if (!['Porn'].includes(className)) {
        return true;
      }
      return probability < PORN_THRESHOLD;
    });

    if (ret) {
      negRets.push({
        filePath,
        predictions: predictions.reduce((acc, cur) => {
          acc[cur.className] = cur.probability;
          return acc;
        }, {})
      });
    }
  });
  await fs.writeFile('./negative-flaw.json', JSON.stringify(negRets, null, 2));

  if (copyFile) {
    for (let neg of negRets) {
      await fs.copyFile(neg.filePath, path.resolve('./temp', path.basename(neg.filePath)));
    }
  }
}


async function createRangeTest(targetDir) {
  const INIT = {
    10: 0,
    20: 0,
    30: 0,
    40: 0,
    50: 0,
    60: 0,
    70: 0,
    80: 0,
    90: 0,
    100: 0,
  };
  const pornRets = {...INIT}
  const hentaiRets = {...INIT}
  const sexyRets = {...INIT}
  const drawRets = {...INIT}
  await test(targetDir, (filePath, predictions) => {
    predictions.forEach(cur => {
      const {className, probability} = cur;
      if (className === 'Porn') {
         for (let key of Object.keys(pornRets)) {
            if (probability * 100 < +key) {
              pornRets[key]++;
              break;
            }
         }
      }
      if (className === 'Hentai') {
        for (let key of Object.keys(hentaiRets)) {
           if (probability * 100 < +key) {
             hentaiRets[key]++;
             break;
           }
        }
     }
     if (className === 'Sexy') {
      for (let key of Object.keys(sexyRets)) {
         if (probability * 100 < +key) {
            sexyRets[key]++;
            break;
         }
      }
     }
     if (className === 'Drawing') {
      for (let key of Object.keys(drawRets)) {
         if (probability * 100 < +key) {
            drawRets[key]++;
            break;
         }
      }
     }
    });
  });
  return {
    pornRets,
    hentaiRets,
    sexyRets,
    drawRets,
  }
}

async function negRangeTest() {
  await fs.writeFile(
    './negative-range.json',
    JSON.stringify(
      await createRangeTest(negDir), null, 2
    )
  );
}


async function sexyRangeTest() {
  await fs.writeFile('./sexy-range.json', 
    JSON.stringify(
      await createRangeTest(sexyDir), null, 2
    )
  );
}


async function normalRangeTest() {
  await fs.writeFile('./normal-range.json',
    JSON.stringify(
      await createRangeTest(rawDir), null, 2
    )
  );
}

load_model().then(async () => {
  await negRangeTest();
  await sexyRangeTest();
  await normalRangeTest();
  await sexyTest();
  await normalTest();
  await negTest();
})
