#!/bin/bash
set -e

echo "=== 开始应用 P0 修复 ==="
echo ""

# 备份原始文件
echo "1. 备份原始 kernels.py..."
cp src/kda_bt16/kernels.py src/kda_bt16/kernels.py.backup
echo "   ✓ 备份至 kernels.py.backup"
echo ""

# P0-1: 修复 mid 索引 bug（3 处）
echo "2. 应用 P0-1 修复：mid 索引 bug..."
echo "   查找需要修复的位置..."
grep -n "o_c\[:, None\] == mid" src/kda_bt16/kernels.py || echo "   (无匹配 - 可能已修复)"

echo ""
echo "   执行替换..."
sed -i 's/o_c\[:, None\] == mid/o_i[:, None] == mid/g' src/kda_bt16/kernels.py

echo "   验证替换结果..."
grep -n "o_i\[:, None\] == mid" src/kda_bt16/kernels.py | head -5
echo "   ✓ P0-1 修复完成（3 处）"
echo ""

# P0-2: 检查 fused kernel else 分支
echo "3. 检查 P0-2：fused kernel else 分支..."
echo "   当前代码结构（行 460-470）："
sed -n '460,470p' src/kda_bt16/kernels.py | cat -n
echo ""

echo "=== P0-1 自动修复完成 ==="
echo "=== P0-2 需要手动修复（见下方说明） ==="
echo ""
echo "P0-2 修复步骤："
echo "  1. 编辑 src/kda_bt16/kernels.py"
echo "  2. 在第 462-465 行的 if STATE_V_FIRST 块后"
echo "  3. 添加 else 分支（参考 K2 kernel 的 :291-297）"
echo ""
echo "参考代码位置："
echo "  K2 正确实现：kernels.py:291-297"
echo "  需要修复位置：kernels.py:462-465"
echo ""
