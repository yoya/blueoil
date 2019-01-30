#ifndef RUNTIME_INCLUDE_TENSORT_HPP_
#define RUNTIME_INCLUDE_TENSORT_HPP_

#include <vector>

template <class T>
class TensorT {
  template<class> friend class TensorT;
  friend class Tensor;
private:
  int calcVolume(const std::vector<int>& shape);
  std::vector<int> shape_;
  std::vector<T> data_;
  int shapeVolume();
public:
  TensorT(std::vector<int> shape);
  TensorT(std::vector<int> shape, std::vector<T> data);
  TensorT(std::vector<int> shape, T *data);
  TensorT(const TensorT<T> &tensor);
  TensorT(const Tensor &tensor);
  Tensor Tensor();
  std::vector<int> shape() const;
  std::vector<T> & data();
  const std::vector<T> & data() const;
  const T *dataAsArray() const;
  T *dataAsArray();
  const T *dataAsArray(std::vector<int> indices) const;
  T *dataAsArray(std::vector<int> indices);
  void dump() const;
  typename std::vector<T>::const_iterator begin() const;
  typename std::vector<T>::const_iterator end() const;
  typename std::vector<T>::iterator begin();
  typename std::vector<T>::iterator end();
  bool allequal(const TensorT<T> &tensor) const;
  bool allclose(const TensorT<T> &tensor) const;
  // rtol: relative tolerance parameter
  // atol: absolute tolerance parameter
  bool allclose(const TensorT<T> &tensor, float rtol, float atol) const;
};

/* copy constructor */
template <class T>
TensorT<T>::TensorT(const class Tensor &tensor)
  : shape_(tensor.shape()),
    data_(std::vector<T>(shapeVolume(std::move(tensor.shape())), 0)) {
  int n = tensor.data().size();
  for (int i = 0 ; i < n ; i++) {
    data_[i] = static_cast<T>(tensor.data()[i]);
  }
}

template <class T>
Tensor TensorT<T>::Tensor() {
  class Tensor t(shape());
  int n = data().size();
  for (int i = 0 ; i < n ; i++) {
    t.data()[i] = static_cast<float>(data()[i]);
  }
  return t;
}

/**/
template <class T>
int TensorT<T>::calcVolume(const std::vector<int>& shape) {
  return std::accumulate(shape.begin(), shape.end(),
			 1, std::multiplies<int>());
}

template <class T>
TensorT<T>::TensorT(std::vector<int> shape)
  : shape_(shape),
    data_(std::vector<T>(calcVolume(std::move(shape)), 0)) {
}

template <class T>
TensorT<T>::TensorT(std::vector<int> shape, std::vector<T> data)
  : shape_(std::move(shape)),
    data_(std::move(data)) {
}

template <class T>
TensorT<T>::TensorT(std::vector<int> shape, T *arr)
  : shape_(shape),
    data_(std::vector<T>(arr,
                         arr + calcVolume(std::move(shape)))) {
}

template <class T>
TensorT<T>::TensorT(const TensorT<T> &tensor)
  : shape_(tensor.shape_),
    data_(tensor.data_) {
}

template <class T>
std::vector<int> TensorT<T>::shape() const {
  return shape_;
}


template <class T>
std::vector<T> &TensorT<T>::data() {
  return data_;
}

template <class T>
const std::vector<T> &TensorT<T>::data() const {
  return data_;
}


template <class T>
const T *TensorT<T>::dataAsArray() const {
  if (shape_.size() == 0) {
    throw std::invalid_argument("Tensor have no shape");
  }
  return data_.data();
}

template <class T>
const T *TensorT<T>::dataAsArray(std::vector<int> indices) const {
  if (shape_.size() != indices.size() ) {
    throw std::invalid_argument("shape.size != indices.size");
  }
  int i = 0;
  for (auto itr = indices.begin(); itr != indices.end(); ++itr, ++i) {
    if ((*itr < 0) || (shape_[i] <= *itr)) {
      throw std::invalid_argument("indices out of shape range");
    }
  }
  int offset = 0, size = data_.size();
  i = 0;
  for (auto itr = indices.begin(); itr != indices.end(); ++itr, ++i) {
    size /= shape_[i];
    offset += (*itr) * size;
  }
  return data_.data() + offset;
}

template <class T>
T *TensorT<T>::dataAsArray() {
  if (shape_.size() == 0) {
    throw std::invalid_argument("Tensor have no shape");
  }
  return data_.data();
}

template <class T>
T *TensorT<T>::dataAsArray(std::vector<int> indices) {
  if (shape_.size() != indices.size() ) {
    throw std::invalid_argument("shape.size != indices.size");
  }
  int i = 0;
  for (auto itr = indices.begin(); itr != indices.end(); ++itr, ++i) {
    if ((*itr < 0) || (shape_[i] <= *itr)) {
      throw std::invalid_argument("indices out of shape range");
    }
  }
  int offset = 0, size = data_.size();
  i = 0;
  for (auto itr = indices.begin(); itr != indices.end(); ++itr, ++i) {
    size /= shape_[i];
    offset += (*itr) * size;
  }
  return data_.data() + offset;
}

#endif  // RUNTIME_INCLUDE_TENSORT_HPP_

