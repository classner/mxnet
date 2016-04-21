/*!
 * Copyright (c) 2016 by Contributors
 * \file unpooling-inl.h
 * \brief
 * \author Christoph Lassner
*/

#ifndef MXNET_OPERATOR_UNPOOLING_INL_H_
#define MXNET_OPERATOR_UNPOOLING_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./guided_pool.h"

namespace mxnet {
namespace op {

namespace unpool_enum {
  enum UnpoolingOpInputs {kData, kPoolData, kPooledData};
  enum UnpoolingOpOutputs {kOut};
  enum UnpoolingOpType {kMaxPooling, kAvgPooling, kSumPooling};
}  // namespace pool_enum

struct UnpoolingParam : public dmlc::Parameter<UnpoolingParam> {
   TShape kernel;
   TShape stride;
   TShape pad;
   int pool_type;
   DMLC_DECLARE_PARAMETER(UnpoolingParam) {
     // TODO(bing) change to only set lower bound
     DMLC_DECLARE_FIELD(kernel)
       .set_expect_ndim(2).enforce_nonzero()
       .describe("pooling kernel size: (y, x)");

     DMLC_DECLARE_FIELD(pool_type)
       .add_enum("max", unpool_enum::kMaxPooling)
       .add_enum("avg", unpool_enum::kAvgPooling)
       .add_enum("sum", unpool_enum::kSumPooling)
       .describe("Pooling type to be applied.");

     int stride_shape[] = {1, 1};
     DMLC_DECLARE_FIELD(stride).set_default(TShape(stride_shape, stride_shape + 2))
       .set_expect_ndim(2).enforce_nonzero()
       .describe("stride: for pooling (y, x)");

     int pad_shape[] = {0, 0};
     DMLC_DECLARE_FIELD(pad).set_default(TShape(pad_shape, pad_shape + 2))
       .set_expect_ndim(2)
       .describe("pad for pooling: (y, x)");
   }
 };

template<typename xpu, typename Reducer>
class UnpoolingOp : public Operator {
 public:
  explicit UnpoolingOp(UnpoolingParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(req.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data = in_data[unpool_enum::kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> pool_data = in_data[unpool_enum::kPoolData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> pooled_data = in_data[unpool_enum::kPooledData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> output_data = out_data[unpool_enum::kOut].get<xpu, 4, real_t>(s);

    mshadow::Shape<2> unpooled_shape = Shape2(pool_data.shape_[2],
                                              pool_data.shape_[3]);

    if (param_.pool_type == unpool_enum::kMaxPooling ||
        param_.pool_type == unpool_enum::kSumPooling) {
      Assign(output_data, req[unpool_enum::kOut],
             crop(unpool<Reducer>(pad(pool_data, param_.pad[0], param_.pad[1]),
                                  pad(pooled_data, 0, 0),
                                  pad(data, 0, 0),
                                  param_.kernel[0],
                                  param_.kernel[1],
                                  param_.stride[0]),
                  unpooled_shape,
                  param_.pad[0],
                  param_.pad[1]));
    } else if (param_.pool_type == unpool_enum::kAvgPooling) {
      Assign(output_data, req[unpool_enum::kOut],
             (1.0f / param_.kernel[0] / param_.kernel[1]) *\
             crop(unpool<Reducer>(pad(pool_data, param_.pad[0], param_.pad[1]),
                                  pad(pooled_data, 0, 0),
                                  pad(data, 0, 0),
                                  param_.kernel[0],
                                  param_.kernel[1],
                                  param_.stride[0]),
                  unpooled_shape,
                  param_.pad[0],
                  param_.pad[1]));
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    // Checks.
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(req.size(), 3);
    CHECK_EQ(in_grad.size(), 3);
    // Create inputs.
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> input_grad = in_grad[0].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> grad = out_grad[unpool_enum::kOut].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> data_pool = in_data[unpool_enum::kPoolData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> data_pooled = in_data[unpool_enum::kPooledData].get<xpu, 4, real_t>(s);

    mshadow::Shape<2> pooled_shape = Shape2(data_pooled.shape_[2],
                                            data_pooled.shape_[3]);

    if (param_.pool_type == unpool_enum::kMaxPooling ||
        param_.pool_type == unpool_enum::kSumPooling) {
      Assign(input_grad, req[unpool_enum::kOut],
             crop(guided_pool<Reducer>(pad(data_pool, param_.pad[0], param_.pad[1]),
                                       pad(data_pooled, 0, 0),
                                       pad(grad, param_.pad[0], param_.pad[1]),
                                       param_.kernel[0],
                                       param_.kernel[1],
                                       param_.stride[0],
                                       param_.pad[0],
                                       param_.pad[1]),
                  pooled_shape,
                  param_.pad[0],
                  param_.pad[1]));
    } else if (param_.pool_type == unpool_enum::kAvgPooling) {
      Assign(input_grad, req[unpool_enum::kData],
             (1.0f / param_.kernel[0] / param_.kernel[1]) *\
             crop(guided_pool<Reducer>(pad(data_pool, param_.pad[0], param_.pad[1]),
                                       pad(data_pooled, 0, 0),
                                       pad(grad, param_.pad[0], param_.pad[1]),
                                       param_.kernel[0],
                                       param_.kernel[1],
                                       param_.stride[0],
                                       param_.pad[0],
                                       param_.pad[1]),
                  pooled_shape,
                  param_.pad[0],
                  param_.pad[1]));
    }
  }

 private:
  UnpoolingParam param_;
};  // class UnpoolingOp

template<typename xpu>
Operator* CreateOp(UnpoolingParam param);


#if DMLC_USE_CXX11
class UnpoolingProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<std::string> ListArguments() const override {
    std::vector<std::string> ret;
    ret.push_back("data");
    ret.push_back("data_pool");
    ret.push_back("data_pooled");
    return ret;
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    CHECK_EQ(in_shape->size(), 3);
    const TShape &dshape = (*in_shape)[0];
    CHECK_EQ(dshape.ndim(), 4) << \
        "Unpooling: Input data shape should be 4D in (batch, channel, y, x)";
    const TShape &pdshape = (*in_shape)[1];
    CHECK_EQ(pdshape.ndim(), 4) << \
      "Unpooling: Pool data shape should be 4D in (batch, channel, y, x)";
    const TShape &pddshape = (*in_shape)[2];
    CHECK_EQ(pddshape.ndim(), 4) << \
      "Unpooling: Pooled data shape should be 4D in (batch, channel, y, x)";
    TShape oshape = pdshape;
    size_t is2 = std::min(pdshape[2] + 2 * param_.pad[0] - param_.kernel[0] + param_.stride[0] - 1,
                          pdshape[2] + 2 * param_.pad[0] - 1) / param_.stride[0] + 1;
    size_t is3 = std::min(pdshape[3] + 2 * param_.pad[1] - param_.kernel[1] + param_.stride[1] - 1,
                          pdshape[3] + 2 * param_.pad[1] - 1) / param_.stride[1] + 1;
    CHECK_EQ(dshape[2], is2) << "Unpooling: differing expected unpool size!";
    CHECK_EQ(dshape[3], is3) << "Unpooling: differing expected unpool size!";
    out_shape->clear();
    out_shape->push_back(oshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    UnpoolingProp *prop_sym = new UnpoolingProp();
    prop_sym->param_ = this->param_;
    return prop_sym;
  }

  std::string TypeString() const override {
    return "Unpooling";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {
        out_grad[0],
        in_data[1],
        in_data[2],
        };
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  UnpoolingParam param_;
};  // class UnpoolingProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_UNPOOLING_INL_H_
