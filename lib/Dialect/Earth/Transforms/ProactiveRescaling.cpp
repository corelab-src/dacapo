
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/Debug.h"

#include "hecate/Dialect/Earth/Analysis/ScaleManagementUnit.h"
#include "hecate/Dialect/Earth/Transforms/Common.h"

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_PROACTIVERESCALING
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct ProactiveRescalingPass
    : public hecate::earth::impl::ProactiveRescalingBase<
          ProactiveRescalingPass> {
  ProactiveRescalingPass() {}

  ProactiveRescalingPass(hecate::earth::ProactiveRescalingOptions ops) {
    this->waterline = ops.waterline;
    this->output_val = ops.output_val;
  }

  void runOnOperation() override {

    auto func = getOperation();

    markAnalysesPreserved<hecate::ScaleManagementUnit>();

    mlir::OpBuilder builder(func);
    mlir::IRRewriter rewriter(builder);
    SmallVector<mlir::Type, 4> inputTypes;

    hecate::earth::refineInputValues(func, builder, inputTypes, waterline,
                                     output_val);
    // Apply waterline rescaling for the operations
    func.walk([&](hecate::earth::ForwardMgmtInterface sop) {
      builder.setInsertionPointAfter(sop.getOperation());
      sop.processOperandsPARS(waterline);
      inferTypeForward(sop);
      sop.processResultsPARS(waterline);
    });
    hecate::earth::refineReturnValues(func, builder, inputTypes, waterline,
                                      output_val);
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
