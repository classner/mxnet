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
  switch (param.pool_type) {
    case unpool_enum::kMaxPooling:
      return new UnpoolingOp<gpu, mshadow::red::maximum>(param);
    case unpool_enum::kAvgPooling:
      return new UnpoolingOp<gpu, mshadow::red::sum>(param);
    case unpool_enum::kSumPooling:
      return new UnpoolingOp<gpu, mshadow::red::sum>(param);
    default:
      LOG(FATAL) << "unknown activation type";
      return NULL;
  }
}

}  // namespace op
}  // namespace mxnet

