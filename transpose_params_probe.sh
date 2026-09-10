#!/bin/bash
for f in /usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/ascendc/include/basic_api/interface/kernel_operator_vec_transpose_intf.h /usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/ascendc/include/basic_api/impl/kernel_operator_vec_transpose_intf_impl.h; do
  echo ===$f===
  grep -n -A45 -B15 'struct TransposeParamsExt\|class TransposeParamsExt\|CheckFunTranspose' "$f" 2>/dev/null | head -180
done
