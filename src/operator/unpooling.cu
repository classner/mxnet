/*!
 * Copyright (c) 2016 by Contributors
 * \file unpooling.cu
 * \brief
 * \author Christoph Lassner
*/

#include "./unpooling-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(UnpoolingParam param) {
  return new UnpoolingOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet

