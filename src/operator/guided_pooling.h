/*!
 *  Copyright (c) 2016 by Contributors
 * \file guided_pool.h
 * \brief support for unpool
 * \author Christoph Lassner
 */
#ifndef MSHADOW_EXTENSION_GUIDED_POOLING_H_
#define MSHADOW_EXTENSION_GUIDED_POOLING_H_
#include <algorithm>
#include <mshadow/extension.h>

namespace mshadow {
namespace expr {
/*!
 * \brief Guided pooling is the reverse expression of guided unpooling.
 * \tparam SrcExp source expression to be pooled from
 * \tparam DType the content data type
 * \tparam srcdim dimension of src
 */
template<typename SrcExp, typename DType, int srcdim>
struct GuidedPoolingExp:
      public MakeTensorExp<GuidedPoolingExp<SrcExp, DType, srcdim>,
                           SrcExp, srcdim, DType> {
  /*! \brief source input, corresponds to src in pooling */
  const SrcExp &data_src_;
  /*! \brief result of pooled data, corresponds to result of pooling */
  const SrcExp &data_pooled_;
  /*! \brief gradient data of unpooled part, to be propagated down */
  const SrcExp &grad_unpooled_;
  /*! \brief shape of pooled expression */
  index_t pshape_y_;
  /*! \brief shape of pooled expression */
  index_t pshape_x_;
  /*! \brief kernel size in height */
  index_t ksize_y_;
  /*! \brief kernel size in width */
  index_t ksize_x_;
  /*! \brief kernel stride */
  index_t kstride_;
  /*! \brief padding x */
  index_t pad_x_;
  /*! \brief padding y */
  index_t pad_y_;
  /*! \brief constructor */
  GuidedPoolingExp(const SrcExp &data_src,
                   const SrcExp &data_pooled,
                   const SrcExp &grad_unpooled,
                   index_t ksize_y, index_t ksize_x, index_t kstride,
                   index_t pad_y, index_t pad_x)
      : data_src_(data_src), data_pooled_(data_pooled),
        grad_unpooled_(grad_unpooled),
        ksize_y_(ksize_y), ksize_x_(ksize_x), kstride_(kstride),
        pad_x_(pad_x), pad_y_(pad_y) {
    Shape<srcdim> upshape = ShapeCheck<srcdim, SrcExp>::Check(grad_unpooled);
    Shape<srcdim> pshape = ShapeCheck<srcdim, SrcExp>::Check(data_pooled);
    Shape<srcdim> sshape = ShapeCheck<srcdim, SrcExp>::Check(data_src);
    for (int k = 0;  k < srcdim - 2; ++k) {
      CHECK_EQ(pshape[k], sshape[k]) << "GuidedPoolingExp: pool and src shape mismatch";
      CHECK_EQ(upshape[k], sshape[k]) << "GuidedPoolingExp: grad and src shape mismatch";
    }
    pshape_x_ = pshape[srcdim - 1];
    pshape_y_ = pshape[srcdim - 2];
    this->shape_ = sshape;
  }
};
/*!
 * \brief Guided pooling gradient for 4D, backprop gradient value back
 * \param data_src  source input, corresponds to src in pooling
 * \param data_pooled result of pooled data, corresponds to result of pooling
 * \param grad_unpooled gradient data of unpooled part, to be propgated down
 * \param ksize_y kernel height
 * \param ksize_x kernel width
 * \param kstride stride for each kernel
 * \return expression corresponding to pooled 4D Tensor, storing backproped gradient
 * \tparam SrcExp source expression
 * \tparam DType the content data type
 * \tparam etype type of expression
 */
template<typename SrcExp, typename DType, int etype>
inline GuidedPoolingExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
guided_pooling(const Exp<SrcExp, DType, etype> &data_src,
               const Exp<SrcExp, DType, etype> &data_pooled,
               const Exp<SrcExp, DType, etype> &grad_unpooled,
               index_t ksize_y, index_t ksize_x, index_t kstride,
               index_t pad_y, index_t pad_x) {
  return GuidedPoolingExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
      (data_src.self(), data_pooled.self(), grad_unpooled.self(),
       ksize_y, ksize_x, kstride, pad_y, pad_x);
}
//----------------------
// Execution plan
//----------------------
template<typename SrcExp, typename DType, int srcdim>
struct Plan<GuidedPoolingExp<SrcExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const GuidedPoolingExp<SrcExp, DType, srcdim> &e)
      : data_src_(MakePlan(e.data_src_)), data_pooled_(MakePlan(e.data_pooled_)),
        grad_unpooled_(MakePlan(e.grad_unpooled_)),
        sshape_y_(e.shape_[srcdim - 2]), sshape_x_(e.shape_[srcdim - 1]),
        pshape_y_(e.pshape_y_),  pshape_x_(e.pshape_x_),
        ksize_y_(e.ksize_y_), ksize_x_(e.ksize_x_),
        pad_y_(e.pad_y_), pad_x_(e.pad_x_), kstride_(e.kstride_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    // There are (at least) two memory saving possibilities to implement this:
    // 1. Using the original unpooled and pooled maps,
    // 2. Using the maps that the unpooling layer received and produced.
    // Since these are available anyways, and its result map then does not
    // necessarily have to be stored, use option 1.
    using namespace std;
    i -= pad_y_;
    j -= pad_x_;
    // i and j are in the unpooled coordinate system, but only on pooled
    // locations.
    const index_t x = j;
    const index_t y = i % sshape_y_;
    const index_t c = i / sshape_y_;
    const DType vsrc = data_pooled_.Eval(c * pshape_y_ + y, x);
    const index_t upy_min = y * kstride_;
    const index_t upx_min = x * kstride_;
    const index_t upy_max = min(y * kstride_ + ksize_y_, sshape_y_);
    const index_t upx_max = min(x * kstride_ + ksize_x_, sshape_x_);

    // Every pooled value originates from the row-major first entry with the
    // maximum  value.
    for (index_t upy = upy_min; upy < upy_max; ++upy) {
      for (index_t upx = upx_min; upx < upx_max; ++upx) {
        if (data_src_.Eval(c * sshape_y_ + upy, upx) == vsrc) {
          return grad_unpooled_.Eval(c * sshape_y_ + upy, upx);
        }
      }
    }
    // This point can not be reached during normal operation. Make the
    // compiler happy.
    return DType(0);
  }

 private:
  Plan<SrcExp, DType> data_src_, data_pooled_, grad_unpooled_;
  const index_t sshape_y_, sshape_x_, pshape_y_, pshape_x_;
  const index_t ksize_y_, ksize_x_, pad_y_, pad_x_;
  const index_t kstride_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_GUIDED_POOLING_H_
