
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "hecate/Support/Support.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "hecate/Dialect/Earth/Analysis/CandidateAnalysis.h"
#include "hecate/Dialect/Earth/Transforms/Common.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include <exception>
#include <random>
#include <thread>

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_BYPASSDETECTION
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;

#define DEBUG_TYPE "dacapo"

namespace {
/// Pass to bufferize Arith ops.
struct BypassDetectionPass
    : public hecate::earth::impl::BypassDetectionBase<BypassDetectionPass> {
  BypassDetectionPass() {}
  BypassDetectionPass(hecate::earth::BypassDetectionOptions ops) {
    this->waterline = ops.waterline;
    this->threshold = ops.threshold;
  }

  void runOnOperation() override {

    /* llvm::errs() << "Bypass Edge Detection\n"; */
    auto func = getOperation();
    auto &ca = getAnalysis<hecate::CandidateAnalysis>();

    // Applying multi-threading to find bypass edges
    auto maxThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> thres;
    for (auto from : ca.getEdges()) {
      auto threadFunc =
          std::bind(&BypassDetectionPass::findBypassEdge, this, func, from);
      thres.emplace_back(threadFunc);
      if (thres.size() >= maxThreads) {
        for (std::thread &th : thres) {
          th.join();
        }
        thres.clear();
      }
    }
    for (std::thread &th : thres) {
      th.join();
    }
    // Organize the validLiveOuts
    for (auto a : ca.getEdges()) {
      auto v = ca.getValueInfo(a);
      mlir::SmallVector<int64_t, 4> validTargets;
      for (auto bp : v->getLiveOuts()) {
        auto vp = ca.getValueInfo(bp);
        if (!vp->isBypassEdge(a)) {
          validTargets.push_back(bp);
        }
      }
      v->setValidLiveOuts(validTargets);
      ca.sortValidCandidates(a);
    }
    markAnalysesPreserved<hecate::CandidateAnalysis>();
  }

  void findBypassEdge(mlir::func::FuncOp func, int64_t from) {
    auto mod = mlir::ModuleOp::create(func.getLoc());
    PassManager pm(mod.getContext());
    pm.addNestedPass<func::FuncOp>(hecate::earth::createBootstrapPlacement());

    auto &ca = getAnalysis<hecate::CandidateAnalysis>();
    auto dup = func.clone();
    auto &&block = dup.getBody().front();
    auto &&operations = block.getOperations();
    mlir::OpBuilder builder(dup);
    dup->setAttr("btp_target", builder.getDenseI64ArrayAttr(
                                   ca.getValueInfo(from)->getLiveOuts()));
    /* builder.getDenseI64ArrayAttr(ca.getTargets(from))); */
    /* dup->setAttr("segment_return", builder.getDenseI64ArrayAttr({})); */
    mod.push_back(dup);

    if (pm.run(mod).failed()) {
      llvm::errs() << "bootstrap placement failed" << '\n';
    }

    for (auto argval : dup.getArguments()) {
      argval.setType(
          argval.getType().dyn_cast<RankedTensorType>().replaceSubElements(
              [&](hecate::earth::HEScaleTypeInterface t) {
                return t.switchScale(waterline);
              }));
    }

    for (auto &&op : operations) {
      if (auto sop = dyn_cast<hecate::earth::ForwardMgmtInterface>(op)) {
        auto opid = hecate::getIntegerAttr("opid", sop->getResult(0));
        if (!isa<hecate::earth::BootstrapOp>(sop) && opid < from)
          continue;

        // PARS Scale Management
        builder.setInsertionPointAfter(sop.getOperation());
        sop.processOperandsPARS(waterline);
        inferTypeForward(sop);
        sop.processResultsPARS(waterline);
        /////////////////////////////////////////

        // check over threshold and set bypass
        if (sop.overThreshold(threshold)) {
          ca.getValueInfo(from)->setThresholdOpid(opid);
          break;
        }
      }
    }
    dup.erase();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
