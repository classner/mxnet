/*!
 *  Copyright (c) 2016 by Contributors
 * \file spatial_unpool.h
 * \brief support for unpool
 * \author Christoph Lassner
 */
#ifndef MSHADOW_EXTENSION_GUIDED_UNPOOLING_H_
#define MSHADOW_EXTENSION_GUIDED_UNPOOLING_H_
#include <algorithm>
#include <mshadow/extension.h>
namespace mshadow {
namespace expr {
/*!
 * \brief The guided unpooling expr reverses the operation of max pooling.
 * \tparam SrcExp source expression to be pooled from.
 * \tparam DType the content data type.
 * \tparam srcdim dimension of src.
 */
template<typename SrcExp, typename DType, int srcdim>
struct GuidedUnpoolingExp:
      public MakeTensorExp<GuidedUnpoolingExp<SrcExp, DType, srcdim>,
                           SrcExp, srcdim, DType> {
  /*! \brief source input, corresponds to src in pooling */
  const SrcExp &data_src_;
  /*! \brief result of pooled data, corresponds to result of pooling */
  const SrcExp &data_pooled_;
  /*! \brief gradient data of pooled part, to be propgate down */
  const SrcExp &grad_pooled_;
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
  /*! \brief padding height */
  index_t pad_y_;
  /*! \brief padding width */
  index_t pad_x_;
  /*! \brief constructor */
  GuidedUnpoolingExp(const SrcExp &data_src,
                     const SrcExp &data_pooled,
                     const SrcExp &grad_pooled,
                     index_t ksize_y, index_t ksize_x, index_t kstride,
                     index_t pad_y, index_t pad_x)
      : data_src_(data_src), data_pooled_(data_pooled),
        grad_pooled_(grad_pooled),
    ksize_y_(ksize_y), ksize_x_(ksize_x), kstride_(kstride),
    pad_y_(pad_y), pad_x_(pad_x) {
    Shape<srcdim> gshape = ShapeCheck<srcdim, SrcExp>::Check(grad_pooled);
    Shape<srcdim> pshape = ShapeCheck<srcdim, SrcExp>::Check(data_pooled);
    Shape<srcdim> sshape = ShapeCheck<srcdim, SrcExp>::Check(data_src);
    for (int k = 0;  k < srcdim - 2; ++k) {
      CHECK_EQ(pshape[k], sshape[k]) << "GuidedUnpoolingExp: pool and src shape mismatch";
      CHECK_EQ(gshape[k], sshape[k]) << "GuidedUnpoolingExp: grad and src shape mismatch";
    }
    pshape_x_ = pshape[srcdim - 1];
    pshape_y_ = pshape[srcdim - 2];
    this->shape_ = sshape;
  }
};
/*!
 * \brief Guided unpooling of 4D data, reversing the operation of max pooling.
 * The operation takes all border cases into account, such as larger kernel
 * size than steps and solves them correctly.
 *
 * \param data_src  source input, corresponds to src in pooling
 * \param data_pooled result of pooled data, corresponds to result of pooling
 * \param grad_pooled gradient data of pooled part, to be propgate down
 * \param ksize_y kernel height
 * \param ksize_x kernel width
 * \param kstride stride for each kernel
 * \return expression corresponding to unpooled 4D Tensor, storing backproped gradient
 * \tparam SrcExp source expression
 * \tparam DType the content data type
 * \tparam etype type of expression
 */
template <typename SrcExp, typename DType, int etype>
inline GuidedUnpoolingExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
guided_unpooling(const Exp<SrcExp, DType, etype> &data_src,
       const Exp<SrcExp, DType, etype> &data_pooled,
       const Exp<SrcExp, DType, etype> &grad_pooled,
                 index_t ksize_y, index_t ksize_x, index_t kstride,
                 index_t pad_y, index_t pad_x) {
  return GuidedUnpoolingExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
      (data_src.self(), data_pooled.self(), grad_pooled.self(),
       ksize_y, ksize_x, kstride, pad_y, pad_x);
}
//----------------------
// Execution plan
//----------------------
template<typename SrcExp, typename DType, int srcdim>
struct Plan<GuidedUnpoolingExp<SrcExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const GuidedUnpoolingExp<SrcExp, DType, srcdim> &e)
      : data_src_(MakePlan(e.data_src_)), data_pooled_(MakePlan(e.data_pooled_)),
        grad_pooled_(MakePlan(e.grad_pooled_)), sshape_y_(e.shape_[srcdim - 2]),
        sshape_x_(e.shape_[srcdim - 1]),
        pshape_y_(e.pshape_y_),  pshape_x_(e.pshape_x_),
    ksize_y_(e.ksize_y_), ksize_x_(e.ksize_x_), kstride_(e.kstride_),
    pad_y_(e.pad_y_), pad_x_(e.pad_x_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    using namespace std;
    const index_t x = j;
    const index_t y = i % sshape_y_;
    const index_t c = i / sshape_y_;
    const DType vsrc = data_src_.Eval(i, j);

    // Determine all possible pooling locations this value may have been used for.
    const index_t py_min =
      y < (ksize_y_) ? 0 : (y - ksize_y_ + kstride_) / kstride_;
    const index_t px_min =
      x < (ksize_x_) ? 0 : (x - ksize_x_ + kstride_) / kstride_;
    const index_t py_max = min((y + kstride_) / kstride_, pshape_y_);
    const index_t px_max = min((x + kstride_) / kstride_, pshape_x_);
    // Use this information to search only for the first max it came
    // from to invert the operation accurately.
    index_t upx_min, upy_min, upx_max, upy_max;
    /*
    printf("pos: %d, %d. py_min: %d, py_max: %d, px_min: %d, px_max: %d.\n",
    y, x, py_min, py_max, px_min, px_max);*/
    DType val = static_cast<DType>(0);
    for (index_t py = py_min; py < py_max; ++py) {
      for (index_t px = px_min; px < px_max; ++px) {
        if (data_pooled_.Eval(c*pshape_y_+py, px) != vsrc) {
          // This point was not responsible for the pooled value.
          continue;
        }
        // We know that the value at the pooled point was the
        // maximum, so higher values can not occur. Stick to the caffe/cudnn
        // implementation using the first row major index.
        index_t upy_pos = py * kstride_;
        index_t upx_pos = px * kstride_;
        upy_min = py * kstride_;
        upx_min = px * kstride_;
        upy_max = min(y + 1, min(upy_pos + ksize_y_, sshape_y_));
        upx_max = min(upx_pos + ksize_x_, sshape_x_);
        /*
        printf("uppos: %d, %d. upy_min: %d, upy_max: %d, upx_min: %d, upx_max: %d.\n",
        upy_pos, upx_pos, upy_min, upy_max, upx_min, upx_max);*/
        // Is this point the largest in the region and may claim the contribution
        // from the pooled point?
        bool use_point = true;
        for (index_t upy = upy_min; upy < upy_max; ++upy) {
          for (index_t upx = upx_min; upx < upx_max; ++upx) {/*
            printf("upsearch: %d, %d. uppos: %d, %d. vsrc: %f. val: %f.\n",
            upy, upx, y, x, vsrc, data_src_.Eval(c * sshape_y_ + upy, upx));*/
            if (upx == x && upy == y) {
              break; // outer loop will be left as well due to upy_max.
            }
            if (data_src_.Eval(c * sshape_y_ + upy, upx) == vsrc) {
              // This maximum has precedence.
              use_point = false;
              break;
            }
          }
          if (! use_point) {
            break;
          }
        }
        if (use_point) {
          // Apply the reduction.
          val += mshadow::red::maximum::PartialGrad(vsrc,
                data_pooled_.Eval(c * pshape_y_ + py, px)) *
            grad_pooled_.Eval(c * pshape_y_ + py, px);
        }
      }
    }
    return val;
  }

 private:
  Plan<SrcExp, DType> data_src_, data_pooled_, grad_pooled_;
  const index_t sshape_y_, sshape_x_, pshape_y_, pshape_x_;
  const index_t ksize_y_, ksize_x_;
  const index_t kstride_;
  const index_t pad_y_, pad_x_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_GUIDED_UNPOOLING_H_
