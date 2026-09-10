#!/bin/bash
f=/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/ascendc/include/basic_api/interface/kernel_operator_vec_transpose_intf.h
f2=/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/ascendc/include/basic_api/impl/kernel_operator_vec_transpose_intf_impl.h
sed -n '35,90p' "$f"
echo '--- impl ---'
sed -n '85,225p' "$f2"
