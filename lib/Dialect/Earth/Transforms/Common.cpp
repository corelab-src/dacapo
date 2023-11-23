#include "hecate/Dialect/Earth/Transforms/Common.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include<iostream>
using namespace mlir;

int64_t hecate::earth::refineLevel(mlir::OpBuilder builder, mlir::Operation *op,
                                int64_t waterline, int64_t output_val,
                                int64_t min_level) {
  int64_t max_required_level =
      hecate::earth::EarthDialect::bootstrapLevelUpperBound - min_level;

  builder.setInsertionPoint(op);

  int64_t acc_scale_max = 0;
  int64_t rescalingFactor = hecate::earth::EarthDialect::rescalingFactor;
  for (auto v : op->getOperands()) {
    auto st = v.getType().dyn_cast<hecate::earth::HEScaleTypeInterface>();
    auto acc_scale = st.getLevel() * rescalingFactor + st.getScale();
    acc_scale_max = std::max(acc_scale_max, acc_scale);
  }

  if(hecate::earth::EarthDialect::bootstrapLevelUpperBound < 3)
  {
    max_required_level =
        (acc_scale_max + output_val + rescalingFactor - 1) / rescalingFactor;
  }
  for (size_t i = 0; i < op->getNumOperands(); i++) {
    auto v = op->getOperand(i);
    auto st = v.getType().dyn_cast<hecate::earth::HEScaleTypeInterface>();
    auto acc_scale = st.getLevel() * rescalingFactor + st.getScale();
    int64_t required_level =
        (acc_scale + output_val + rescalingFactor - 1) / rescalingFactor;
    int64_t level_diff = max_required_level - required_level;
    //auto max_acc_scale = max_required_level * rescalingFactor;
    //int64_t level_diff = (max_acc_scale - acc_scale) / rescalingFactor;


    op->setOperand(i, builder.create<hecate::earth::ModswitchOp>(
                          op->getLoc(), v, level_diff));
  }
  return max_required_level;
}

void hecate::earth::refineReturnValues(mlir::func::FuncOp func,
                                       mlir::OpBuilder builder,
                                       SmallVector<mlir::Type, 4> inputTypes,
                                       int64_t waterline, int64_t output_val) {

  int64_t max_required_level;
  //    hecate::earth::EarthDialect::bootstrapLevelUpperBound;
  // Reduce the level of the resulting values to reduce the size of returns
  //

  auto rop = dyn_cast<func::ReturnOp>(func.getBlocks().front().getTerminator());
  max_required_level = hecate::earth::refineLevel(builder, rop, waterline, output_val, 0);
  bool first_bootstrap = false;
  func.walk([&](hecate::earth::BootstrapOp bop) {
    auto max_level = hecate::earth::refineLevel(
        builder, bop, waterline, output_val,
        hecate::earth::EarthDialect::bootstrapLevelLowerBound - 1);
    if(!first_bootstrap) {
    	max_required_level = max_level;
	    first_bootstrap = true;
    }
  });
  
  /* TODO : It is temporary max_required_level setting. It should be fixed. */
  /* The max level of first bootstrap is not correct. (Not compiled) */
  if(hecate::earth::EarthDialect::bootstrapLevelUpperBound >= 3)
  {
    max_required_level = hecate::earth::EarthDialect::bootstrapLevelUpperBound;
  }

  /* func.walk([&](func::ReturnOp rop) { */
  // Remap the return types
  func.setFunctionType(
      builder.getFunctionType(inputTypes, rop.getOperandTypes()));
  /* }); */
  func->setAttr("init_level", builder.getI64IntegerAttr(max_required_level));
  
  SmallVector<int64_t, 4> scales_in;
  SmallVector<int64_t, 4> scales_out;

  for (auto &&arg : func.getArguments()) {
    scales_in.push_back(arg.getType()
                            .dyn_cast<hecate::earth::HEScaleTypeInterface>()
                            .getScale());
  }
  func->setAttr("arg_scale", builder.getDenseI64ArrayAttr(scales_in));
  for (auto &&restype : func.getResultTypes()) {
    scales_out.push_back(
        restype.dyn_cast<hecate::earth::HEScaleTypeInterface>().getScale());
  }
  func->setAttr("res_scale", builder.getDenseI64ArrayAttr(scales_out));
}

void hecate::earth::inferTypeForward(hecate::earth::ForwardMgmtInterface sop) {
  Operation *oop = sop.getOperation();
  auto iop = dyn_cast<mlir::InferTypeOpInterface>(oop);
  SmallVector<Type, 4> retTypes;
  if (iop.inferReturnTypes(oop->getContext(), oop->getLoc(), oop->getOperands(),
                           oop->getAttrDictionary(), oop->getRegions(),
                           retTypes)
          .succeeded()) {
    oop->getResults().back().setType(retTypes.back());
  }
}
