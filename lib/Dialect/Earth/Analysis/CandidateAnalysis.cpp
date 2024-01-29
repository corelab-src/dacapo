#include "hecate/Dialect/Earth/Analysis/CandidateAnalysis.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Support/Support.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace hecate;

hecate::CandidateAnalysis::CandidateAnalysis(mlir::Operation *op)
    : _l(op), _op(op), smu(op) {

  llvm::SmallVector<int64_t, 4> liveIn, liveOut;
  smu.attach();
  values.emplace_back(0, nullptr);
  edges.push_back(0);
  _op->walk([&](hecate::earth::HEScaleOpInterface sop) {
    if ((llvm::isa<hecate::earth::UpscaleOp>(sop) ||
         llvm::isa<hecate::earth::RescaleOp>(sop) ||
         llvm::isa<hecate::earth::BootstrapOp>(sop) ||
         llvm::isa<hecate::earth::ModswitchOp>(sop))) {
      assert(0 && "Currently not supported");
    }
    int64_t opid = values.size();
    for (auto &&val : sop.getOperation()->getResults()) {
      values.emplace_back(opid, val);
      hecate::setIntegerAttr("opid", val, opid);
    }
    if (!sop.isCipher())
      return;
    for (auto &&oper : sop->getOperands()) {
      if (!hecate::earth::getScaleType(oper).isCipher())
        continue;
      if (_l.isDeadAfter(oper, sop)) {
        auto operID = hecate::getIntegerAttr("opid", oper);
        auto dead = std::find(liveOut.begin(), liveOut.end(), operID);
        if (dead != liveOut.end()) {
          liveOut.erase(dead);
          values[operID].setDeadOpid(opid);
        }
      }
    }
    liveOut.push_back(opid);
    std::map<std::pair<int64_t, int64_t>, mlir::OpOperand *> edgeMap;
    for (auto &&user : sop->getUsers()) {
      if (smu.getID(user) != smu.getID(sop) && opid > 10) {
        values[opid].setLiveOuts(liveOut);
        values[opid].setLiveIns(liveIn);
        edges.push_back(opid);
        break;
      }
    }
    liveIn = liveOut;
  });
  retOpid = values.size();
  values.emplace_back(retOpid, nullptr);
  toFromMap[0] = {};
}

size_t hecate::CandidateAnalysis::getNumValues() const { return values.size(); }
int64_t hecate::CandidateAnalysis::getRetOpid() const { return retOpid; }
size_t hecate::CandidateAnalysis::getNumEdges() const { return edges.size(); }

size_t hecate::CandidateAnalysis::getMaxNumOuts() const {
  size_t ret = 0;
  for (auto a : candidateSet) {
    if (a.first > ret)
      ret = a.first;
  }
  return ret;
}
mlir::SmallVector<int64_t, 4> hecate::CandidateAnalysis::getEdges() const {
  return edges;
}
mlir::SmallVector<int64_t, 4> hecate::CandidateAnalysis::getCandidates() const {
  return candidates;
}
mlir::SmallVector<int64_t, 4>
hecate::CandidateAnalysis::getCandidateSet(size_t size) const {
  auto &&set = candidateSet.find(size);
  if (set != candidateSet.end())
    return set->second;
  else
    return {};
}
mlir::SmallVector<int64_t, 4>
hecate::CandidateAnalysis::getTargets(int64_t opid) const {
  if (opid == retOpid)
    return {};
  if (!values[opid].getValidLiveOuts().size())
    return values[opid].getLiveOuts();
  else
    return values[opid].getValidLiveOuts();
}
mlir::SmallVector<int64_t, 4>
hecate::CandidateAnalysis::getTargets(int64_t opid, int64_t setNum) const {
  if (opid == retOpid)
    return {};
  if (values[opid].getValidLiveOuts().size() == setNum)
    return values[opid].getValidLiveOuts();
  else
    return values[opid].getLiveOuts();
}
ValueInfo *hecate::CandidateAnalysis::getValueInfo(int64_t opid) {
  return &values[opid];
}

void hecate::CandidateAnalysis::sortValidCandidates(int64_t opid) {
  auto v = values[opid];
  candidateSet[v.getValidLiveOuts().size()].push_back(opid);
  if (v.getLiveOuts().size() != v.getValidLiveOuts().size()) {
    candidateSet[v.getLiveOuts().size()].push_back(opid);
  }
  return;
}
mlir::SmallVector<int64_t, 4>
hecate::CandidateAnalysis::sortTargets(int64_t setNum) {
  mlir::SmallVector<int64_t, 4> sortedTargets;
  for (int i = 1; i <= setNum; i++) {
    for (auto a : candidateSet[setNum]) {
      for (auto b : getTargets(a, setNum)) {
        auto &&itr = std::find(sortedTargets.begin(), sortedTargets.end(), b);
        if (itr == sortedTargets.end()) {
          sortedTargets.push_back(b);
        }
      }
    }
  }
  return sortedTargets;
}

mlir::SmallVector<int64_t, 4>
hecate::CandidateAnalysis::sortTargets(int64_t setNum,
                                       mlir::SmallVector<int64_t, 4> opids) {
  mlir::SmallVector<int64_t, 4> sortedTargets;
  for (auto a : opids) {
    for (auto b : getTargets(a, setNum)) {
      auto &&itr = std::find(sortedTargets.begin(), sortedTargets.end(), b);
      if (itr == sortedTargets.end()) {
        sortedTargets.push_back(b);
      }
    }
  }
  return sortedTargets;
}

void hecate::CandidateAnalysis::finalizeCandidates(int64_t setNum) {
  candidates.push_back(0);
  /* llvm::errs() << "setNum " << setNum << '\n'; */
  for (int i = 1; i <= setNum; i++) {
    /* for (auto a : getCandidateSet(i)) { */
    /*   llvm::errs() << a << " "; */
    /* } */
    candidates.append(getCandidateSet(i));
  }
  candidates.push_back(getRetOpid());
  std::sort(candidates.begin(), candidates.end());
  return;
}

void hecate::CandidateAnalysis::pushFromCoverage(
    int64_t from, mlir::SmallVector<int64_t, 2> coverages) {
  auto c = coverages.front();
  auto bc = coverages.back();
  if (c < 0)
    c = getRetOpid();
  if (bc < 0)
    bc = getRetOpid();
  for (auto to : candidates) {
    if (to < bc && from < to) {
      toFromMap[to].push_back(from);
    } else if (to == getRetOpid() && c == getRetOpid()) {
      toFromMap[to].push_back(from);
    }
  }
  return;
}

hecate::ValueInfo::ValueInfo(int64_t opid, mlir::Value v) : opid(opid), v(v) {}
void hecate::ValueInfo::setLiveOuts(mlir::SmallVector<int64_t, 4> outs) {
  liveOuts = outs;
  return;
}

void hecate::ValueInfo::setLiveIns(mlir::SmallVector<int64_t, 4> ins) {
  liveIns = ins;
  return;
}
void hecate::ValueInfo::setCoverage(int64_t c) {
  coverage = c;
  return;
}
void hecate::ValueInfo::setBootCoverage(int64_t bc) {
  bootCoverage = bc;
  return;
}
void hecate::ValueInfo::setDeadOpid(int64_t dead) {
  deadOpid = dead;
  return;
}
void hecate::ValueInfo::setBypassEdge(int64_t overThr) {
  if (!v)
    return;
  if (deadOpid > overThr) {
    isBypass = true;
    hecate::setIntegerAttr("is_bypassed", v, 1);
    return;
  }
  return;
}
void hecate::ValueInfo::setValidLiveOuts(mlir::SmallVector<int64_t, 4> outs) {
  validLiveOuts = outs;
  return;
}

mlir::Value hecate::ValueInfo::getValue() const { return v; }
mlir::SmallVector<int64_t, 4> hecate::ValueInfo::getLiveIns() const {
  return liveIns;
}
mlir::SmallVector<int64_t, 4> hecate::ValueInfo::getLiveOuts() const {
  return liveOuts;
}
int64_t hecate::ValueInfo::getCoverage() const { return coverage; }
bool hecate::ValueInfo::isBypassEdge() const { return isBypass; }
int64_t hecate::ValueInfo::getBootCoverage() const { return bootCoverage; }
int64_t hecate::ValueInfo::getDeadOpid() const { return deadOpid; }
mlir::SmallVector<int64_t, 4> hecate::ValueInfo::getValidLiveOuts() const {
  return validLiveOuts;
}
