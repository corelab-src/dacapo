
#include "hecate/Dialect/Earth/Analysis/CandidateAnalysis.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "hecate/Dialect/Earth/Analysis/CandidateAnalysis.h"
/* #include "hecate/Dialect/Earth/Analysis/ScaleManagementUnit.h" */
#include "hecate/Dialect/Earth/Transforms/Common.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include <random>

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_DACAPOPLANNER
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;

#define DEBUG_TYPE "dacapo"

namespace {
/// Pass to bufferize Arith ops.
struct DaCapoPlannerPass
    : public hecate::earth::impl::DaCapoPlannerBase<DaCapoPlannerPass> {
  DaCapoPlannerPass() {}
  DaCapoPlannerPass(hecate::earth::DaCapoPlannerOptions ops) {}
  DaCapoPlannerPass(std::pair<int64_t, int64_t> ops) {
    this->waterline = ops.first;
    this->output_val = ops.second;
  }

  void runOnOperation() override {

    auto func = getOperation();
    mlir::OpBuilder builder(func);

    auto &ca = getAnalysis<hecate::CandidateAnalysis>();

    // to, bestplan{latency, cutted_edges, return_type}
    DenseMap<int64_t, std::tuple<double, SmallVector<int64_t, 4>,
                                 SmallVector<Type, 4>, mlir::func::FuncOp>>
        bestPlan;

    SmallVector<mlir::Type, 4> inputTypes;
    for (auto argval : func.getArguments()) {
      inputTypes.push_back(mlir::RankedTensorType::get(
          llvm::SmallVector<int64_t, 1>{1},
          builder.getType<hecate::earth::CipherType>(waterline, 0)));
    }
    bestPlan[0] = {0.0, {}, inputTypes, {}};

    auto mod = mlir::ModuleOp::create(func.getLoc());

    // pm : PassManager_BootstrappingPlanner
    // pmC : PassManager_CoverageRecorder
    PassManager pm(mod.getContext()), pmC(mod.getContext());
    // add check partitioning, and separate passmanager,
    pm.addNestedPass<func::FuncOp>(hecate::earth::createBootstrapPlacement());
    pm.addNestedPass<func::FuncOp>(hecate::earth::createCodeSegmentation());
    pm.addNestedPass<func::FuncOp>(
        hecate::earth::createProactiveRescaling({waterline, output_val}));
    pm.addNestedPass<func::FuncOp>(hecate::earth::createEarlyModswitch());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addNestedPass<func::FuncOp>(hecate::earth::createLatencyEstimator());

    pmC.addNestedPass<func::FuncOp>(hecate::earth::createBootstrapPlacement());
    pmC.addNestedPass<func::FuncOp>(hecate::earth::createCodeSegmentation());
    pmC.addNestedPass<func::FuncOp>(
        hecate::earth::createCoverageRecorder({waterline, 0.5}));

    int64_t setNum =
        func->getAttrOfType<mlir::IntegerAttr>("selected_set").getInt();
    for (auto to : ca.getCandidates()) {
      double optCost = std::numeric_limits<double>::max();
      func::FuncOp optFunc;
      for (auto from : ca.toFromMap[to]) {
        auto dup = func.clone();
        dup.setName((func.getName() + "_" + std::to_string(from) + "_" +
                     std::to_string(to))
                        .str());
        dup->setAttr("cutted_edge", builder.getDenseI64ArrayAttr({from, to}));
        dup->setAttr("btp_target",
                     builder.getDenseI64ArrayAttr(ca.getTargets(from, setNum)));
        dup->setAttr(
            "segment_input",
            builder.getDenseI64ArrayAttr(ca.getValueInfo(from)->getLiveOuts()));
        dup->setAttr("segment_inputType",
                     builder.getTypeArrayAttr(std::get<2>(bestPlan[from])));
        dup->setAttr("segment_return", builder.getDenseI64ArrayAttr(
                                           ca.getValueInfo(to)->getLiveOuts()));
        dup->setAttr("is_mid_segment", builder.getBoolAttr(true));

        mod.push_back(dup);
        if (pm.run(mod).failed()) {
          llvm::errs() << "pm Pass failed" << '\n';
          dup.dump();
          assert(0 && "Pass failed inside DaCapo explorer");
        }
        double cost = dup->getAttrOfType<mlir::FloatAttr>("est_latency")
                          .getValueAsDouble() +
                      std::get<0>(bestPlan[from]);
        if (cost < optCost) {
          auto plan = std::get<1>(bestPlan[from]);
          plan.push_back(to);
          llvm::SmallVector<Type, 4> retTypes;
          retTypes.append(inputTypes);
          retTypes.append(mlir::SmallVector<Type, 4>(dup.getResultTypes()));

          bestPlan[to] = {cost, plan, retTypes, dup.clone()};
          optCost = cost;
        }
        dup.erase();
      }

      // calculate the coverage [to] as [from]
      if (to != ca.getRetOpid()) {
        auto dup = func.clone();
        dup->setAttr("cutted_edge",
                     builder.getDenseI64ArrayAttr({to, ca.getRetOpid()}));
        dup->setAttr("btp_target",
                     builder.getDenseI64ArrayAttr(ca.getTargets(to, setNum)));
        dup->setAttr("segment_input", builder.getDenseI64ArrayAttr(
                                          ca.getValueInfo(to)->getLiveOuts()));
        dup->setAttr("segment_inputType",
                     builder.getTypeArrayAttr(std::get<2>(bestPlan[to])));
        dup->setAttr("segment_return", builder.getDenseI64ArrayAttr({}));

        mod.push_back(dup);
        if (pmC.run(mod).failed()) {
          llvm::errs() << "pmC Pass failed" << '\n';
          dup.dump();
          assert(0 && "Pass failed inside DaCapo explorer");
        }
        auto &&coverages =
            dup->getAttrOfType<mlir::DenseI64ArrayAttr>("coverages")
                .asArrayRef();
        ca.pushFromCoverage(to, mlir::SmallVector<int64_t, 2>(coverages));
        dup.erase();
      }

      /* llvm::errs() << '\n'; */
      /* llvm::errs() << "-------------BestPlan Until " << to << '\n'; */
      /* llvm::errs() << " latency : " << std::get<0>(bestPlan[to]) << '\n'; */
      /* llvm::errs() << " cutted Edges : "; */
      /* for (auto a : std::get<1>(bestPlan[to])) */
      /*   llvm::errs() << a << " "; */
      /* llvm::errs() << '\n'; */
      /* llvm::errs() << " to retType : "; */
      /* for (auto a : std::get<2>(bestPlan[to])) { */
      /*   llvm::errs() << a << " "; */
      /* } */
      /* llvm::errs() << '\n'; */

      /* llvm::errs() << '\n'; */
      /* llvm::errs() << "--------------------------\n"; */
      /* if (to > 0) { */
      /*   std::get<3>(bestPlan[to]).dump(); */
      /*   llvm::errs() << '\n'; */
      /*   llvm::errs() << '\n'; */
      /*   optFunc.erase(); */
      /* } */
    }
    auto targets =
        ca.sortTargets(setNum, std::get<1>(bestPlan[ca.getRetOpid()]));
    /* llvm::errs() << "Estimated Latency : " */
    /*              << std::get<0>(bestPlan[ca.getRetOpid()]) << '\n'; */
    /* llvm::errs() << " FINAL Bootstrapping Target : " << targets.size() <<
     * '\n'; */
    /* std::get<3>(bestPlan[ca.getRetOpid()]).dump(); */
    /* llvm::errs() << "--------------------------\n"; */
    /* for (auto dd : targets) */
    /*   llvm::errs() << dd << " "; */
    /* llvm::errs() << '\n'; */
    func->setAttr("btp_target", builder.getDenseI64ArrayAttr(targets));
    markAnalysesPreserved<hecate::CandidateAnalysis>();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
