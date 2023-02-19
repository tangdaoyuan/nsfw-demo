import jpeg from 'jpeg-js';
import Sharp from 'sharp';

export const decodeRealityImage = function (imageType, image) {
  return new Promise((resolve, reject) => {
    Sharp(image)
      .modulate({
        saturation: 0.7
      })
      .blur(2)
      .resize({ width: 512 })
      .toFormat('jpeg')
      .toBuffer({ resolveWithObject: true })
      .then(({ data, info }) => {
        const imageData = jpeg.decode(data, true);
        resolve(imageData);
      })
      .catch(err => { reject(err) });
  });
}

export const decodeAnimeImage = function (imageType, image) {
  return new Promise((resolve, reject) => {
    Sharp(image)
      .resize({ width: 512 })
      .toFormat('jpeg')
      .toBuffer({ resolveWithObject: true })
      .then(({ data, info }) => {
        const imageData = jpeg.decode(data, true);
        resolve(imageData);
      })
      .catch(err => { reject(err) });
  });
}
