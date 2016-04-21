/*!
 * Copyright (c) 2016 by Contributors
 * \file unpooling.cc
 * \brief
 * \author Christoph Lassner
*/
#include "./unpooling-inl.h"

namespace mxnet {
namespace op {

  template<>
  Operator *CreateOp<cpu>(UnpoolingParam param) {
    switch (param.pool_type) {
    case unpool_enum::kMaxPooling:
      return new UnpoolingOp<cpu, mshadow::red::maximum>(param);
    case unpool_enum::kAvgPooling:
      return new UnpoolingOp<cpu, mshadow::red::sum>(param);
    case unpool_enum::kSumPooling:
      return new UnpoolingOp<cpu, mshadow::red::sum>(param);
    default:
      LOG(FATAL) << "unknown activation type";
      return NULL;
    }
  }

  DMLC_REGISTER_PARAMETER(UnpoolingParam);

Operator* UnpoolingProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

MXNET_REGISTER_OP_PROPERTY(Unpooling, UnpoolingProp)
.describe("Perform spatial unpooling on inputs.")
.add_argument("data", "Symbol[]", "Input data to the pooling operator.")
.add_arguments(UnpoolingParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

