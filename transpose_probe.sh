#!/bin/bash
for root in /usr/local/Ascend/cann/x86_64-linux/ascendc/include /usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/ascendc/include; do
  if [ -d "$root" ]; then
    grep -R -n -E '(^|[^A-Za-z])Transpose[[:space:]]*\(' "$root" --include='*.h' --include='*.hpp' 2>/dev/null | head -60
  fi
done
