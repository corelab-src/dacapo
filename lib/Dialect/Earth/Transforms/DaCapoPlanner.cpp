
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
#include "llvm/Support/Format.h"
#include <random>

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_DACAPOPLANNER
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;

/* #define DEBUG_TYPE "dacapo" */
#define DEBUG_TYPE "Debug"

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
    DenseMap<int64_t,
             std::tuple<double, SmallVector<int64_t, 4>, SmallVector<Type, 4>,
                        mlir::func::FuncOp, SmallVector<bool, 2>>>
        bestPlan;
    DenseMap<int64_t, Type> bypassInputTypes;

    SmallVector<mlir::Type, 4> inputTypes;
    SmallVector<bool, 2> inputBypasses;
    for (auto argval : func.getArguments()) {
      auto tp = mlir::RankedTensorType::get(
          llvm::SmallVector<int64_t, 1>{1},
          builder.getType<hecate::earth::CipherType>(waterline, 0));
      /* bypassInputTypes[inputTypes.size()] = tp; */
      inputTypes.push_back(tp);
      inputBypasses.push_back(true);
    }
    bestPlan[0] = {0.0, {}, inputTypes, {}, inputBypasses};

    auto mod = mlir::ModuleOp::create(func.getLoc());

    // pm : PassManager_BootstrappingPlanner
    // pmC : PassManager_CoverageRecorder
    PassManager pm(mod.getContext()), pmC(mod.getContext());
    // add check partitioning, and separate passmanager,
    pm.addNestedPass<func::FuncOp>(hecate::earth::createBootstrapPlacement());
    pm.addNestedPass<func::FuncOp>(hecate::earth::createCodeSegmentation());
    pm.addNestedPass<func::FuncOp>(
        hecate::earth::createProactiveRescaling({waterline, output_val}));
    /* pm.addNestedPass<func::FuncOp>(hecate::earth::createUpscaleBubbling());
     */
    /* pm.addPass(mlir::createCanonicalizerPass()); */
    /* pm.addNestedPass<func::FuncOp>( */
    /*     hecate::earth::createWaterlineRescaling({waterline, output_val})); */
    /* pm.addNestedPass<func::FuncOp>( */
    /*     hecate::earth::createSNRRescaling({waterline, output_val})); */
    pm.addNestedPass<func::FuncOp>(hecate::earth::createEarlyModswitch());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    /* pm.addNestedPass<func::FuncOp>(hecate::earth::createFlexibleBootstrap());
     */
    pm.addNestedPass<func::FuncOp>(hecate::earth::createLatencyEstimator());

    pmC.addNestedPass<func::FuncOp>(hecate::earth::createBootstrapPlacement());
    pmC.addNestedPass<func::FuncOp>(hecate::earth::createCodeSegmentation());
    /* pmC.addNestedPass<func::FuncOp>( */
    /*     hecate::earth::createLocalBypassDetection()); */
    pmC.addNestedPass<func::FuncOp>(
        hecate::earth::createCoverageRecorder({waterline, 0.5}));
    /* llvm::dbgs() << "Return Opid : " << ca.getRetOpid() << '\n'; */

    int64_t setNum =
        func->getAttrOfType<mlir::IntegerAttr>("selected_set").getInt();
    for (auto to : ca.getCandidates()) {
      double optCost = std::numeric_limits<double>::max();
      func::FuncOp optFunc;
      LLVM_DEBUG(llvm::dbgs() << to << " s btp_targets: ";
                 for (auto bbbb
                      : ca.getTargets(to, setNum)) llvm::dbgs()
                 << bbbb << " ";
                 llvm::dbgs() << '\n';);
      for (auto from : ca.toFromMap[to]) {
        auto vif = ca.getValueInfo(from);
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

        dup->setAttr("segment_bypassType",
                     builder.getBoolArrayAttr(std::get<4>(bestPlan[from])));
        dup->setAttr("segment_returnBypasses",
                     builder.getBoolArrayAttr(ca.getBypassTypeOfLiveOuts(to)));
        mod.push_back(dup);
        if (pm.run(mod).failed()) {
          llvm::dbgs() << "pm Pass failed" << '\n';
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
          llvm::SmallVector<bool, 2> bypassTypes;
          bypassTypes.append(inputBypasses);
          bypassTypes.append(ca.getBypassTypeOfLiveOuts(to));
          bestPlan[to] = {cost, plan, retTypes, dup.clone(), bypassTypes};
          optCost = cost;
        }
        dup.erase();
      }

      // calculate the coverage [to] as [from]
      if (to != ca.getRetOpid()) {
        auto dup = func.clone();
        dup->setAttr("cutted_edge",
                     builder.getDenseI64ArrayAttr({to, ca.getRetOpid()}));
        /* dup->setAttr("btp_target", */
        /*              builder.getDenseI64ArrayAttr(ca.getTargets(to,
         * setNum))); */
        dup->setAttr("btp_target",
                     builder.getDenseI64ArrayAttr(ca.getTargets(to, setNum)));
        dup->setAttr("segment_input", builder.getDenseI64ArrayAttr(
                                          ca.getValueInfo(to)->getLiveOuts()));
        dup->setAttr("segment_inputType",
                     builder.getTypeArrayAttr(std::get<2>(bestPlan[to])));
        dup->setAttr("segment_return", builder.getDenseI64ArrayAttr({}));
        dup->setAttr("segment_inputBypasses",
                     builder.getBoolArrayAttr(std::get<4>(bestPlan[to])));
        dup->setAttr("segment_returnBypasses", builder.getBoolArrayAttr({}));
        /* builder.getBoolArrayAttr(ca.getBypassTypeOfLiveOuts(to))); */
        /* llvm::dbgs() << "segment bypassType [" << to << "]:"; */
        /* for (auto aa : ca.getBypassTypeOfLiveOuts(to)) { */

        // print Bypass Type
        /* for (auto aa : std::get<4>(bestPlan[to])) { */
        /*   llvm::dbgs() << aa << " "; */
        /* } */
        /* llvm::dbgs() << '\n'; */

        mod.push_back(dup);
        if (pmC.run(mod).failed()) {
          llvm::dbgs() << "pmC Pass failed" << '\n';
          dup.dump();
          assert(0 && "Pass failed inside DaCapo explorer");
        }
        auto &&coverages =
            dup->getAttrOfType<mlir::DenseI64ArrayAttr>("coverages")
                .asArrayRef();
        ca.pushFromCoverage(to, mlir::SmallVector<int64_t, 2>(coverages));
        dup.erase();
      }
      /* LLVM_DEBUG( */
      llvm::dbgs() << '\n';
      llvm::dbgs() << "-------------BestPlan Until " << to << '\n';
      llvm::dbgs() << " latency : " << std::get<0>(bestPlan[to]) << '\n';
      llvm::dbgs() << " cutted Edges : ";
      for (auto a : std::get<1>(bestPlan[to]))
        llvm::dbgs() << a << " ";
      llvm::dbgs() << '\n';
      llvm::dbgs() << " to retType : ";
      for (auto a : std::get<2>(bestPlan[to])) {
        llvm::dbgs() << a << " ";
      }
      llvm::dbgs() << '\n';
      llvm::dbgs() << " to liveouts bypass type : ";
      for (auto a : std::get<4>(bestPlan[to])) {
        llvm::dbgs() << a << " ";
      }
      llvm::dbgs() << '\n';

      llvm::dbgs() << '\n';
      llvm::dbgs() << "--------------------------\n";
      if (to > 0) {
        /* std::get<3>(bestPlan[to]).dump(); */
        llvm::dbgs() << '\n';
        llvm::dbgs() << '\n';
      }
      /* ); */
    }

    auto targets =
        ca.sortTargets(setNum, std::get<1>(bestPlan[ca.getRetOpid()]));

    llvm::outs() << llvm::format("Estimated Latency : %lf (sec) \n",
                                 std::get<0>(bestPlan[ca.getRetOpid()]) /
                                     1000000);
    llvm::outs() << "Number of Bootstrapping : " << targets.size() << '\n';

    /* LLVM_DEBUG( */
    llvm::dbgs() << "Estimated Latency : "
                 << std::get<0>(bestPlan[ca.getRetOpid()]) << '\n';
    /* std::get<3>(bestPlan[ca.getRetOpid()]).dump(); */
    llvm::dbgs() << " FINAL Bootstrapping Target : " << targets.size() << '\n';
    llvm::dbgs() << "RetOpid : " << ca.getRetOpid() << '\n';
    llvm::dbgs() << "--------------------------\n";
    for (auto dd : std::get<1>(bestPlan[ca.getRetOpid()]))
      llvm::dbgs() << dd << " ";
    llvm::dbgs() << '\n';

    /* ); */
    func->setAttr("btp_target", builder.getDenseI64ArrayAttr(targets));
    markAnalysesPreserved<hecate::CandidateAnalysis>();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
