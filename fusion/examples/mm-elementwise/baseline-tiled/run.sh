#!/bin/bash
WORKLOAD=mm-elementwise-baseline-tiled
if [[ "$1" == "--gpu" ]]
then
  BIN=./${WORKLOAD}.gpu.bin
  TRACE=trace.gpu.out
else
  BIN=./${WORKLOAD}.bin
  TRACE=trace.cpu.out
fi
if [[ "$2" == "--prof" ]]
then
  PROF="nsys profile -w true -t cuda,osrt,cudnn,cublas -s none -o ${WORKLOAD} -f true -x true"
else
  PROF=
fi
if [[ "$1" == "--gpu" ]]
then
  shift
fi
if [[ "$1" == "--prof" ]]
then
  shift
fi
${PROF} ${BIN} "$@" > ${TRACE} ; grep WARN ${TRACE}
if [ -n "${PROF}" ]
then
  nsys stats --force-export --force-overwrite --format csv --output . --report cudaapisum,cudaapitrace,gpumemsizesum,gpumemtimesum,gputrace,gpukernsum ${WORKLOAD}.nsys-rep
  grep Kernel ${WORKLOAD}_gputrace.csv
fi
