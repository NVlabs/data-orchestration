#!/bin/bash
WORKLOAD=mm-elementwise-baseline
if [[ "$1" == "--gpu" ]]
then
  BIN=./${WORKLOAD}.gpu.bin
  TRACE=trace.gpu.out
else
  BIN=./${WORKLOAD}.bin
  TRACE=trace.cpu.out
fi
${BIN} > ${TRACE} ; grep WARN ${TRACE}
