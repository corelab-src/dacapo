
#include "hecate/Dialect/Earth/Analysis/CandidateAnalysis.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include <fstream>

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_CANDIDATESELECTION
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;
#define DEBUG_TYPE "Debug"

namespace {
/// Pass to bufferize Arith ops.
struct CandidateSelectionPass
    : public hecate::earth::impl::CandidateSelectionBase<
          CandidateSelectionPass> {
  CandidateSelectionPass() {}
  CandidateSelectionPass(hecate::earth::CandidateSelectionOptions ops) {
    this->waterline = ops.waterline;
    this->output_val = ops.output_val;
  }

  void runOnOperation() override {

    auto func = getOperation();
    auto &ca = getAnalysis<hecate::CandidateAnalysis>();

    auto mod = mlir::ModuleOp::create(func.getLoc());
    PassManager pm(mod.getContext());
    pm.addNestedPass<func::FuncOp>(hecate::earth::createBootstrapPlacement());
    pm.addNestedPass<func::FuncOp>(
        hecate::earth::createProactiveRescaling({waterline, output_val}));

    for (size_t i = 1; i < ca.getMaxNumOuts(); i++) {
      auto dup = func.clone();
      mlir::OpBuilder builder(dup);
      dup->setAttr("btp_target",
                   builder.getDenseI64ArrayAttr(ca.sortTargets(i)));
      mod.push_back(dup);
      if (pm.run(mod).succeeded()) {
        func->setAttr("selected_set", builder.getI64IntegerAttr(i));
        LLVM_DEBUG(llvm::dbgs() << "selected _set : " << i << '\n';);
        ca.finalizeCandidates(i);
        dup.erase();
        break;
      }
      dup.erase();
    }

    markAnalysesPreserved<hecate::CandidateAnalysis>();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
