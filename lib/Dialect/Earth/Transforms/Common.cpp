#include "hecate/Dialect/Earth/Transforms/Common.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Support/Support.h"
#include "mlir/IR/Value.h"

using namespace mlir;

void hecate::earth::refineLevel(mlir::OpBuilder builder, mlir::Operation *op,
                                int64_t waterline, int64_t output_val,
                                int64_t min_level) {
  int64_t max_required_level =
      hecate::earth::EarthDialect::bootstrapLevelUpperBound - min_level;
  if (max_required_level < 0) {
    max_required_level =
        hecate::earth::EarthDialect::levelUpperBound - min_level;
  }

  builder.setInsertionPoint(op);

  int64_t acc_scale_max = 0;
  int64_t rescalingFactor = hecate::earth::EarthDialect::rescalingFactor;
  for (auto v : op->getOperands()) {
    auto st = v.getType().dyn_cast<hecate::earth::HEScaleTypeInterface>();
    auto acc_scale = st.getLevel() * rescalingFactor + st.getScale();
    acc_scale_max = std::max(acc_scale_max, acc_scale);
  }
  for (size_t i = 0; i < op->getNumOperands(); i++) {
    auto v = op->getOperand(i);
    if (hecate::getIntegerAttr("is_bypassed", v) > 0) {
      /* llvm::errs() << "pass early modswitch \n"; */
      /* v.dump(); */
      continue;
    }
    auto st = v.getType().dyn_cast<hecate::earth::HEScaleTypeInterface>();
    auto acc_scale =
        st.getLevel() * rescalingFactor + st.getScale() + output_val;
    auto max_acc_scale = max_required_level * rescalingFactor;
    int64_t level_diff = (max_acc_scale - acc_scale) / rescalingFactor;
    op->setOperand(i, builder.create<hecate::earth::ModswitchOp>(
                          op->getLoc(), v, level_diff));
  }
}

void hecate::earth::refineReturnValues(mlir::func::FuncOp func,
                                       mlir::OpBuilder builder,
                                       SmallVector<mlir::Type, 4> inputTypes,
                                       int64_t waterline, int64_t output_val) {

  int64_t max_required_level =
      hecate::earth::EarthDialect::bootstrapLevelUpperBound;
  if (max_required_level < 0)
    max_required_level = hecate::earth::EarthDialect::levelUpperBound;

  // Reduce the level of the resulting values to reduce the size of returns
  //
  auto rop = dyn_cast<func::ReturnOp>(func.getBlocks().front().getTerminator());

  if (func->hasAttr("is_mid_segment") &&
      func->getAttrOfType<mlir::BoolAttr>("is_mid_segment").getValue()) {
    auto &&bypassTypes = func->getAttr("segment_returnBypasses")
                             .dyn_cast<mlir::ArrayAttr>()
                             .getValue();
    for (size_t i = 0; i < rop->getNumOperands(); i++) {
      auto v = rop->getOperand(i);
      bool isBypass = bypassTypes[i].dyn_cast<mlir::BoolAttr>().getValue();
      hecate::setIntegerAttr("is_bypassed", v, isBypass);
    }
    hecate::earth::refineLevel(
        builder, rop, waterline, 0,
        hecate::earth::EarthDialect::bootstrapLevelLowerBound - 1);

  } else
    hecate::earth::refineLevel(builder, rop, waterline, output_val, 0);

  func.walk([&](hecate::earth::BootstrapOp bop) {
    hecate::earth::refineLevel(
        builder, bop, waterline, 0,
        hecate::earth::EarthDialect::bootstrapLevelLowerBound - 1);
  });

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

/* llvm::SmallVector<mlir::Type, 4> */
void hecate::earth::refineInputValues(mlir::func::FuncOp func,
                                      mlir::OpBuilder builder,
                                      SmallVector<mlir::Type, 4> &inputTypes,
                                      int64_t waterline, int64_t output_val) {
  // Set function argument types
  if (!func->hasAttr("segment_inputType")) {
    for (auto argval : func.getArguments()) {
      argval.setType(
          argval.getType().dyn_cast<RankedTensorType>().replaceSubElements(
              [&](hecate::earth::HEScaleTypeInterface t) {
                return t.switchScale(waterline);
              }));
      inputTypes.push_back(argval.getType());
    }
  } else {
    auto &&inputType_attrs = func->getAttr("segment_inputType")
                                 .dyn_cast<mlir::ArrayAttr>()
                                 .getValue();
    for (size_t i = 0; i < func.getNumArguments(); i++) {
      auto argval = func.getArgument(i);
      auto input_type = inputType_attrs[i]
                            .dyn_cast<mlir::TypeAttr>()
                            .getValue()
                            .dyn_cast<hecate::earth::HEScaleTypeInterface>();
      argval.setType(input_type);
      inputTypes.push_back(argval.getType());
    }
  }
  return;
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

llvm::SmallVector<mlir::Value, 4>
hecate::earth::attachOpid(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::Value, 4> values;
  // attach the opid to operation
  values.push_back(NULL);
  func->walk([&](hecate::earth::HEScaleOpInterface sop) {
    if ((llvm::isa<hecate::earth::UpscaleOp>(sop) ||
         llvm::isa<hecate::earth::RescaleOp>(sop) ||
         llvm::isa<hecate::earth::BootstrapOp>(sop) ||
         llvm::isa<hecate::earth::ModswitchOp>(sop))) {
      /* assert(0 && "Currently not supported"); */
      return;
    }
    for (auto &&val : sop.getOperation()->getResults()) {
      values.push_back(val);
    }
  });
  return values;
}
