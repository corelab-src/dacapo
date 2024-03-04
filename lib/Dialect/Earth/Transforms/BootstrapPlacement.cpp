

#include "hecate/Dialect/Earth/Analysis/CandidateAnalysis.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Common.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/Debug.h"
#include <fstream>
#include <random>

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_BOOTSTRAPPLACEMENT
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate
  //
#define DEBUG_TYPE "dacapo"

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct BootstrapPlacementPass
    : public hecate::earth::impl::BootstrapPlacementBase<
          BootstrapPlacementPass> {
  BootstrapPlacementPass() {}
  void runOnOperation() override {
    auto func = getOperation();
    mlir::OpBuilder builder(func);
    mlir::IRRewriter rewriter(builder);
    auto values = hecate::earth::attachOpid(func);
    auto &ca = getAnalysis<hecate::CandidateAnalysis>();
    auto &&btp_target =
        func->getAttrOfType<mlir::DenseI64ArrayAttr>("btp_target").asArrayRef();
    // Bootstrapping Placement based on btp_targets
    for (auto opid : btp_target) {
      if (opid == ca.getRetOpid())
        continue;
      auto &&target = values[opid];
      if (target.getType()
              .dyn_cast<hecate::earth::HEScaleTypeInterface>()
              .isCipher()) {
        builder.setInsertionPointAfterValue(target);
        auto old = target;
        auto btp =
            builder.create<hecate::earth::BootstrapOp>(target.getLoc(), target);
        rewriter.replaceAllUsesExcept(old, btp, btp);
        hecate::setIntegerAttr("opid", btp, opid);
      }
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
