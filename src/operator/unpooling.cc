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
    return new UnpoolingOp<cpu>(param);
  }

  DMLC_REGISTER_PARAMETER(UnpoolingParam);

  Operator* UnpoolingProp::CreateOperator(Context ctx) const {
    DO_BIND_DISPATCH(CreateOp, param_);
  }

  MXNET_REGISTER_OP_PROPERTY(Unpooling, UnpoolingProp)
  .describe("Perform spatial unpooling on inputs.")
  .add_argument("data", "Symbol", "Input data to the guided unpooling operator.")
  .add_argument("data_unpooled", "Symbol", "Input data to the pooling operation to invert.")
  .add_argument("data_pooled", "Symbol", "The pooled data from the pooling operation to invert.")
  .add_arguments(UnpoolingParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
