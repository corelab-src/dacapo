
#ifndef HECATE_DIALECT_EARTH_TRANSFORMS_PASSES
#define HECATE_DIALECT_EARTH_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def ElideConstant: Pass<"elide-constant", "::mlir::func::FuncOp"> {
  let summary = "Elide Constants";
  let description = [{
    This pass saves the constant datas in a {function name}.cst file.
      The file contains the list of the (length, data) pair.
      The constant value will be changed to the index.
  }];
  let options = [
    Option<"name", "name", "std::string", /*default=*/[{""}],
    "Name of output file">
  ];
}
def PrivatizeConstant: Pass<"privatize-constant", "::mlir::func::FuncOp"> {
  let summary = "Privatize Constants";
  let description = [{
    All of the constant op should have a single use after this pass.
  }];
}
def WaterlineRescaling : Pass<"waterline-rescaling", "::mlir::func::FuncOp"> {
  let summary = "Apply Waterline Rescaling";
  let description = [{
    This pass implements the waterline rescaling of EVA.
  }];
  let options = [
    Option<"waterline", "waterline", "int64_t", /*default=*/"20",
    "The minimal result scale of rescaling">,
    Option<"output_val", "output_val", "int64_t", /*default=*/"10",
    "The maximum result value bits of return value">,
  ];
}
def SNRRescaling : Pass<"snr-rescaling", "::mlir::func::FuncOp"> {
  let summary = "Apply SNR Rescaling";
  let description = [{
    This pass implements the SNR rescaling of ELASM.
  }];
  let options = [
    Option<"waterline", "waterline", "int64_t", /*default=*/"20",
    "The minimal result scale of rescaling">,
    Option<"output_val", "output_val", "int64_t", /*default=*/"10",
    "The maximum result value bits of return value">,
  ];
}
def ProactiveRescaling : Pass<"proactive-rescaling", "::mlir::func::FuncOp"> {
  let summary = "Apply Proactive Rescaling";
  let description = [{
    This pass implements the proactive rescaling of Hecate.
  }];
  let options = [
    Option<"waterline", "waterline", "int64_t", /*default=*/"20",
    "The minimal result scale of rescaling">,
    Option<"output_val", "output_val", "int64_t", /*default=*/"10",
    "The maximum result value bits of return value">,
  ];
}
def EarlyModswitch: Pass<"early-modswitch", "::mlir::func::FuncOp"> {
  let summary = "Apply Early Modswitch";
  let description = [{
    This pass implements the early modswitch of EVA.
  }];
}
def UpscaleBubbling : Pass<"upscale-bubbling", "::mlir::func::FuncOp"> {
  let summary = "Apply Upscale Bubbling";
  let description = [{
    This pass may accelerate the convergence of ELASM.
      This pass is not proposed on the ELASM paper.
  }];
}
def SMUChecker : Pass<"check-smu", "::mlir::func::FuncOp"> {
  let summary = "Check the correctness of SMU generation";
  let description = [{
    This pass checks the correctness of SMU generation.
  }];
}
def SMUEmbedding : Pass<"embed-smu", "::mlir::func::FuncOp"> {
  let summary = "Embed SMU IDs to the operation";
  let description = [{
    This pass embeds the SMU analysis. Embdding analysis means that the SMU analysis of copy of the function can be recovered.
  }];
}
def ScaleManagementScheduler : Pass<"schedule-scale-management", "::mlir::func::FuncOp"> {
  let summary = "Schedule the scale management with apply_schedule op";
  let description = [{
    This pass schedules the scale management with apply_schedule op  }];
}
def ELASMExplorer : Pass<"elasm-explorer", "::mlir::func::FuncOp"> {
  let summary = "Error-latency-aware scale management driver pass";
  let description = [{
    This pass iteratively generates schedule, applies schedule, generates code, evaluate the error and latency.}];
  let options = [
    Option<"waterline", "waterline", "int64_t", /*default=*/"20",
    "The minimal result scale of rescaling">,
    Option<"output_val", "output_val", "int64_t", /*default=*/"10",
    "The maximum result value bits of return value">,
    Option<"parallel", "parallel", "int64_t", /*default=*/"20",
    "The number of parallel explorer">,
    Option<"num_iter", "num_iter", "int64_t", /*default=*/"1000",
    "The iteration count of explorer">,
    Option<"beta", "beta", "int64_t", /*default=*/"50",
    "The minimal result scale of rescaling">,
    Option<"gamma", "gamma", "int64_t", /*default=*/"50",
    "The minimal result scale of rescaling">,
  ];
}
def ErrorEstimator : Pass<"estimate-error", "::mlir::func::FuncOp"> {
  let summary = "Estimate the error of a given function";
  let description = [{
    This pass estimates the resulting error of function and attaches attribute.
      This pass requires for function to fully scale managed and all operations have HEScaleOpInterface,
           HEProfInterface, and HEAutoDiffInterface}];
}
def LatencyEstimator : Pass<"estimate-latency", "::mlir::func::FuncOp"> {
  let summary = "Estimate the latency of a given function";
  let description = [{
    This pass estimates the overall latency of function and attaches attribute.
      This pass requires for function to fully scale managed and all operations have HEScaleOpInterface,
           HEProfInterface}];
}
def RemoveBootstrap : Pass<"remove-bootstrap", "::mlir::func::FuncOp"> {
  let summary = "Remove manually inserted bootstrap";
  let description = [{
    This pass removes manually inserted bootstrapping operation.
  }];
}
def FlexibleBootstrap : Pass<"flexible-bootstrap", "::mlir::func::FuncOp"> {
  let summary = "Change bootstrapping target level as required";
  let description = [{
    This pass changes the target level of bootstrapping.
      This pass optimizes the bootstrapping latency by lowering level as required.
  }];
}
def DaCapoPlanner : Pass<"dacapo-planner", "::mlir::func::FuncOp"> {
  let summary = "Automatic bootstrapping placement driver pass";
  let description = [{
    This pass iteratively generates code segements, excepts bypass edges,
         makes bootstrapping placement plan, generates code, evaluate the latency.}];
  let options = [
    Option<"waterline", "waterline", "int64_t", /*default=*/"40",
    "The minimal result scale of rescaling">,
    Option<"output_val", "output_val", "int64_t", /*default=*/"10",
    "The maximum result value bits of return value">,
  ];
}

def BootstrapPlacement : Pass<"bootstrapping-placement", "::mlir::func::FuncOp"> {
  let summary = "Placement bootstrapping operationst based on target edges";
  let description = [{
    This pass places the bootstrapping operations to target edges.
  }];
}

def CoverageRecorder : Pass<"coverage-recording", "::mlir::func::FuncOp"> {
  let summary = "Recording Coverage";
  let description = [{
    This pass records the coverage of each operation.
      The coverage means that the reacheable point(op) starting with bootstrapping.
  }];
  let options = [
    Option<"waterline", "waterline", "int64_t", /*default=*/"40",
    "The minimal result scale of rescaling">,
    Option<"threshold", "threshold", "float", /*default=*/"0.5",
    "The maximum result value bits of return value">,
  ];
}

def BypassDetection : Pass<"bypass-detection", "::mlir::func::FuncOp"> {
  let summary = "Detect Bypasses";
  let description = [{
    This pass checks whether the edge is bypassed-edge or not.
  }];
  let options = [
    Option<"waterline", "waterline", "int64_t", /*default=*/"40",
    "The minimal result scale of rescaling">,
    Option<"threshold", "threshold", "float", /*default=*/"0.5",
    "The maximum result value bits of return value">,
  ];
}
def CandidateSelection : Pass<"candidate-selection", "::mlir::func::FuncOp"> {
  let summary = "Select Candidates";
  let description = [{
    This pass selects candidates based on candidate analysis
  }];
  let options = [
    Option<"waterline", "waterline", "int64_t", /*default=*/"40",
    "The minimal result scale of rescaling">,
    Option<"output_val", "output_val", "int64_t", /*default=*/"10",
    "The maximum result value bits of return value">,
  ];


}
def CodeSegmentation : Pass<"code-segmentation", "::mlir::func::FuncOp"> {
  let summary = "Partition code using start, end edges info";
  let description = [{
    This pass make code segments by cutting edges of code .
  }];
}
/* def ScaleManagementStructuring : Pass<> */


#endif
