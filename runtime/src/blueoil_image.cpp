#include <iostream>
#include <cmath>
#include <cassert>

#include "blueoil.hpp"
#include "blueoil_image.hpp"


namespace blueoil {
namespace image {


template <typename T>
T clamp(const T x, const T lowerLimit, const T upperLimit) {
  if (x < lowerLimit) {
    return lowerLimit;
  }
  if (upperLimit < x) {
    return upperLimit;
  }
  return x;
}

/*
 * Resize Image (Nearest Neighbor)
 */
template <class T>
TensorT<T> ResizeHorizontal_NearestNeighbor(const TensorT<T> &tensor, const int width) {
  auto shape = tensor.shape();
  const int srcHeight = shape[0];
  const int srcWidth  = shape[1];
  const int channels  = shape[2];
  const int height = srcHeight;
  TensorT<T> dstTensor({height, width, channels});
  int srcRGBlinesize = srcWidth * channels;
  float xScale = static_cast<float>(width) / static_cast<float>(srcWidth);
  float srcRGBscaled = 1.0f / xScale;
  const T *srcImageData = tensor.dataAsArray();
  T *srcRGBline = const_cast<T *>(srcImageData);
  T *dstRGB = dstTensor.dataAsArray();
  for (int dstY = 0 ; dstY < height ; dstY++) {
    float srcRGBindexF = 0;
    for (int dstX = 0 ; dstX < width ; dstX++) {
      T *srcRGB = srcRGBline + (static_cast<int>(srcRGBindexF) * channels);
      for (int c = 0 ; c < channels ; c++) {
        *dstRGB++ = *srcRGB++;
      }
      srcRGBindexF += srcRGBscaled;
    }
    srcRGBline += srcRGBlinesize;
  }
  //return TensorT<T>::Tensor(dstTensor);
  return dstTensor;
}

template <class T>
TensorT<T> ResizeVertical_NearestNeighbor(const TensorT<T> &tensor, const int height) {
  auto shape = tensor.shape();
  const int srcHeight = shape[0];
  const int srcWidth  = shape[1];
  const int channels  = shape[2];
  const int width = srcWidth;
  TensorT<T> dstTensor({height, width, channels});
  const int srcScanLineSize = width * channels;
  float yScale = static_cast<float> (height) / static_cast<float>(srcHeight);
  float srcRGBscaled = 1.0f / yScale;
  const T *srcImageData = tensor.dataAsArray();
  T *srcRGBbase = const_cast<T *>(srcImageData);
  T *dstRGB = dstTensor.dataAsArray();
  float srcRGBindexF = 0;
  for (int dstY = 0 ; dstY < height ; dstY++) {
    T *srcRGB = srcRGBbase + (static_cast<int>(srcRGBindexF) * srcScanLineSize);
    for (int i = 0 ; i < srcScanLineSize ; i++) {
      *dstRGB++ = *srcRGB++;
    }
    srcRGBindexF += srcRGBscaled;
  }
  return dstTensor;
  // return TensorT<T>::Tensor(dstTensor);
}

/*
 * Resize Image (Bi-Linear)
 */
template <class T>
TensorT<T> ResizeHorizontal_BiLinear(const TensorT<T> &tensor, const int width) {
  auto shape = tensor.shape();
  const int srcHeight = shape[0];
  const int srcWidth  = shape[1];
  const int channels  = shape[2];
  const int height = srcHeight;
  TensorT<T> dstTensor({height, width, channels});
  float xScale = static_cast<float>(width) / static_cast<float>(srcWidth);
  int xSrcWindow = std::floor(1/xScale);
  xSrcWindow = (xSrcWindow < 2)? 2 :xSrcWindow;
  for (int dstY = 0 ; dstY < height ; dstY++) {
    for (int dstX = 0 ; dstX < width ; dstX++) {
      int srcX = (int) std::floor(dstX/xScale);
      int srcY = dstY;
      for (int c = 0 ; c < channels ; c++) {
        int v = 0.0;
        int totalW = 0.0;
        for (int x = -xSrcWindow ; x < xSrcWindow; x++){
          int srcX2 = clamp(srcX + x, 0, srcWidth - 1);
          const auto *srcRGB = tensor.dataAsArray({srcY, srcX2, 0});
          float d = std::abs(static_cast<float>(x) / static_cast<float> (xSrcWindow));
          float w = 1.0 - d; // Bi-Linear
          v += w * srcRGB[c];
          totalW += w;
        }
        auto *dstRGB = dstTensor.dataAsArray({dstY, dstX, 0});
        dstRGB[c] = v / totalW;
      }
    }
  }
  return dstTensor;
  // return TensorT<T>::Tensor(dstTensor);
}

template <class T>
TensorT<T> ResizeVertical_BiLinear(const TensorT<T> &tensor, const int height) {
  auto shape = tensor.shape();
  const int srcHeight = shape[0];
  const int srcWidth  = shape[1];
  const int channels  = shape[2];
  const int width = srcWidth;
  TensorT<uint8_t> dstTensor({height, width, channels});
  float yScale = static_cast<float> (height) / static_cast<float>(srcHeight);
  int ySrcWindow = std::floor(1/yScale);
  ySrcWindow = (ySrcWindow < 2)? 2 :ySrcWindow;
  for (int dstY = 0 ; dstY < height ; dstY++) {
    for (int dstX = 0 ; dstX < width ; dstX++) {
      int srcX = dstX;
      int srcY = (int) std::floor(dstY/yScale);
      for (int c = 0 ; c < channels ; c++) {
        int v = 0.0;
        int totalW = 0.0;
        for (int y = -ySrcWindow ; y < ySrcWindow ; y++) {
          int srcY2 = clamp(srcY + y, 0, srcHeight - 1);
          const auto *srcRGB = tensor.dataAsArray({srcY2, srcX, 0});
          float d = std::abs(static_cast<float>(y) / static_cast<float> (ySrcWindow));
          float w = 1.0 - d; // Bi-Linear
          v += w * srcRGB[c];
          totalW += w;
        }
        auto *dstRGB = dstTensor.dataAsArray({dstY, dstX, 0});
        dstRGB[c] = v / totalW;
      }
    }
  }
  return dstTensor;
}

Tensor Resize(const Tensor& image, const int width, const int height,
              const enum ResizeFilter filter) {
  auto shape = image.shape();
  int channels = shape[2];
  assert(shape.size() == 3); // 3D shape: HWC
  assert((channels == 1) || (channels == 3)); // grayscale or RGB
  assert((filter == RESIZE_FILTER_NEAREST_NEIGHBOR) || (channels == RESIZE_FILTER_BI_LINEAR));
  const int srcHeight = shape[0];
  const int srcWidth  = shape[1];
  TensorT<uint8_t> tmpImage(image.shape());
  int tmpImageDataSize = tmpImage.data().size();
  float *srcData = const_cast<float *>(image.dataAsArray());
  uint8_t *tmpData = tmpImage.dataAsArray();
  for (int i = 0 ; i < tmpImageDataSize ; i++) {
    *tmpData++ = static_cast<uint8_t>(*srcData++);
  }
  if  (srcWidth != width) {
    if (filter == RESIZE_FILTER_NEAREST_NEIGHBOR) {
      tmpImage = ResizeHorizontal_NearestNeighbor(tmpImage, width);
    } else {  // RESIZE_FILTER_BI_LINEAR
      tmpImage = ResizeHorizontal_BiLinear(tmpImage, width);
    }
  }
  if  (srcHeight != height) {
    if (filter == RESIZE_FILTER_NEAREST_NEIGHBOR) {
      tmpImage = ResizeVertical_NearestNeighbor(tmpImage, height);
    } else {  // RESIZE_FILTER_BI_LINEAR
      tmpImage = ResizeVertical_BiLinear(tmpImage, height);
    }
  }
  //
  Tensor dstImage(tmpImage.shape());
  int dstImageDataSize = dstImage.data().size();
  tmpData = tmpImage.dataAsArray();
  float *dstData = dstImage.dataAsArray();
  for (int i = 0 ; i < dstImageDataSize ; i++) {
    *dstData++ = static_cast<float>(*tmpData++);
  }
  return dstImage;
}   


} // namespace image
} // namespace blueoil
