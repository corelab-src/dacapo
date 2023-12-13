export HECATE=$( cd -- "$( dirname -- "$BASH_SOURCE[0]" )" &> /dev/null && pwd )

alias hopt=$HECATE/build/bin/hecate-opt
alias hopt-debug=$HECATE/build-debug/bin/hecate-opt

mkdir -p $HECATE/examples/traced
mkdir -p $HECATE/examples/optimized/eva
mkdir -p $HECATE/examples/optimized/elasm
mkdir -p $HECATE/examples/optimized/dacapo

build-hopt()(
cd $HECATE/build
ninja
)

build-hoptd()(
cd $HECATE/build-debug
ninja
)

hc-trace()(
cd $HECATE/examples
python3 $HECATE/examples/benchmarks/$1.py
)

hc-test()(
cd $HECATE/examples
python3 $HECATE/examples/tests/$3.py $1 $2 $4 $5
)


hopt-print(){
hopt --$1 --ckks-config="$HECATE/config.json" --waterline=$2 --enable-debug-printer $HECATE/examples/traced/$3.mlir --mlir-print-debuginfo --mlir-pretty-debuginfo --mlir-print-local-scope --mlir-timing -o $HECATE/examples/optimized/$1/$3.$2.mlir
}

hopt-debug-print(){
hopt-debug --$1 --ckks-config="$HECATE/config.json" --waterline=$2 --enable-debug-printer $HECATE/examples/traced/$3.mlir --mlir-print-debuginfo --mlir-pretty-debuginfo --mlir-print-local-scope --mlir-disable-threading  --mlir-timing --mlir-print-ir-after-failure -o $HECATE/examples/optimized/$1/$3.$2.mlir
}

hopt-debug-print-all(){
hopt-debug --$1 --ckks-config="$HECATE/config.json" --waterline=$2 --enable-debug-printer $HECATE/examples/traced/$3.mlir --mlir-print-debuginfo --mlir-pretty-debuginfo --mlir-print-local-scope --mlir-disable-threading  --mlir-timing --mlir-print-ir-after-failure --debug -o $HECATE/examples/optimized/$1/$3.$2.mlir
}

hopt-timing-only(){
hopt --$1 --ckks-config="$HECATE/config.json" --waterline=$2 $HECATE/examples/traced/$3.mlir --mlir-timing -o $HECATE/examples/optimized/$1/$3.$2.mlir
}

hopt-silent(){
hopt --$1 --ckks-config="$HECATE/config.json" --waterline=$2 $HECATE/examples/traced/$3.mlir -o $HECATE/examples/optimized/$1/$3.$2.mlir
}

hc-opt-test() {
hopt-silent $1 $2 $3 && hc-test $1 $2 $3
}

hc-opt-test-timing() {
hopt-timing-only $1 $2 $3 && hc-test $1 $2 $3
}

hopt-heaan-cpu() {
hopt --$1 --ckks-config="$HECATE/profiled_heaan_cpu.json" --waterline=$2 $HECATE/examples/traced/$3.mlir -o $HECATE/examples/optimized/$1/$3.$2.mlir
}

hopt-heaan-gpu() {
hopt --$1 --ckks-config="$HECATE/profiled_heaan_gpu.json" --waterline=$2 $HECATE/examples/traced/$3.mlir -o $HECATE/examples/optimized/$1/$3.$2.mlir
}

hopt-seal() {
hopt --$1 --ckks-config="$HECATE/profiled_SEAL.json" --waterline=$2 $HECATE/examples/traced/$3.mlir -o $HECATE/examples/optimized/$1/$3.$2.mlir
}

hopt-lib-hw() {
hopt --$1 --ckks-config="$HECATE/profiled_$4_$5.json" --waterline=$2 --enable-debug-printer $HECATE/examples/traced/$3.mlir --mlir-print-debuginfo --mlir-pretty-debuginfo --mlir-print-local-scope --mlir-disable-threading --mlir-timing --mlir-print-ir-after-failure -o $HECATE/examples/optimized/$1/$3.$2.mlir
}

hc-back-opt-test(){
hopt-lib-hw $1 $2 $3 $4 $5 && hc-test $1 $2 $3 $4 $5
}

alias hopts-heaan-cpu=hopt-heaan-cpu
alias hopts-heaan-gpu=hopt-heaan-gpu
alias hopts-seal=hopt-seal
alias hbcot=hc-back-opt-test

alias hoptd=hopt-debug-print
alias hopta=hopt-debug-print-all
alias hopts=hopt-silent
alias hoptt=hopt-timing-only
alias hoptp=hopt-print
alias hcot=hc-opt-test
alias hcott=hc-opt-test-timing
