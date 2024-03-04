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
  /* if (values[opid].getValidLiveOuts().size() == setNum) */
  /*   return values[opid].getValidLiveOuts(); */
  /* else */
  /*   return values[opid].getLiveOuts(); */
  if (values[opid].getLiveOuts().size() == setNum)
    return values[opid].getLiveOuts();
  else
    return values[opid].getValidLiveOuts();
}
mlir::SmallVector<int64_t, 4>
hecate::CandidateAnalysis::getTargets(int64_t from, int64_t to,
                                      int64_t setNum) const {
  if (from == retOpid)
    return {};
  if (values[from].getValidLiveOuts().size() == setNum) {
    mlir::SmallVector<int64_t, 4> segLiveOuts = values[from].getValidLiveOuts();
    for (auto tt : values[from].getLiveOuts()) {
      if (!values[tt].isBypassEdge(from))
        continue;
      for (auto user : values[tt].getValue().getUsers()) {
        auto useOpid = hecate::getIntegerAttr("opid", user->getResult(0));
        if (useOpid > from && useOpid < to) {
          segLiveOuts.push_back(tt);
          break;
        }
      }
    }
    /* return values[from].getValidLiveOuts(); */
    return segLiveOuts;
  } else
    return values[from].getLiveOuts();
}
mlir::SmallVector<bool, 4>
hecate::CandidateAnalysis::getBypassTypeOfLiveOuts(int64_t opid) {
  mlir::SmallVector<bool, 4> bypassTypeOfLiveOuts;
  for (auto tt : values[opid].getLiveOuts()) {
    bool bb = values[tt].isBypassEdge(opid);
    bypassTypeOfLiveOuts.push_back(bb);
    /* hecate::setIntegerAttr("is_bypassed", getValueInfo(tt)->getValue(), bb);
     */
  }
  return bypassTypeOfLiveOuts;
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
  /* llvm::errs() << "final targets\n"; */
  for (auto a : opids) {
    /* llvm::errs() << a << " : "; */
    for (auto b : getTargets(a, setNum)) {
      /* llvm::errs() << b << " "; */
      auto &&itr = std::find(sortedTargets.begin(), sortedTargets.end(), b);
      if (itr == sortedTargets.end()) {
        sortedTargets.push_back(b);
      }
    }
    /* llvm::errs() << '\n'; */
  }
  return sortedTargets;
}

void hecate::CandidateAnalysis::finalizeCandidates(int64_t setNum) {
  candidates.push_back(0);
  for (int i = 1; i <= setNum; i++) {
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
  getValueInfo(from)->setCoverage(c);
  getValueInfo(from)->setBootCoverage(bc);
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

void hecate::CandidateAnalysis::setBypassEdges(int64_t from, int64_t overThr) {
  auto v = getValueInfo(from);
  auto deadOpid = v->getDeadOpid();
  if (deadOpid > overThr) {
    getValueInfo(from)->setBypassEdge();
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
void hecate::ValueInfo::setThresholdOpid(int64_t overThr) {
  thresholdOpid = overThr;
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
void hecate::ValueInfo::setBypassEdge() {
  if (!v)
    return;
  isBypass = true;
  hecate::setIntegerAttr("is_bypassed", v, 1);
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
bool hecate::ValueInfo::isBypassEdge(int64_t to) const {
  if (thresholdOpid <= to)
    return true;
  bool isBypass = true;
  for (auto user : getValue().getUsers()) {
    if (auto sop = dyn_cast<hecate::earth::HEScaleOpInterface>(user)) {
      auto useOpid = hecate::getIntegerAttr("opid", sop->getResult(0));
      if (useOpid < thresholdOpid) {
        if (to < useOpid)
          isBypass = false;
      }
    }
  }
  return isBypass;
}
int64_t hecate::ValueInfo::getThresholdOpid() const { return thresholdOpid; }
int64_t hecate::ValueInfo::getBootCoverage() const { return bootCoverage; }
int64_t hecate::ValueInfo::getDeadOpid() const { return deadOpid; }
mlir::SmallVector<int64_t, 4> hecate::ValueInfo::getValidLiveOuts() const {
  return validLiveOuts;
}
