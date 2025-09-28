original_dir=$(pwd)
script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

# echo "Original directory: $original_dir"
# echo "Script directory: $script_dir"

# 创建必要的目录
# mkdir -p ck_tile/include

# # 创建 Composable Kernel 符号链接
# if [ -d "$script_dir/../composable_kernel_fyz" ]; then
#     ln -sf $script_dir/../composable_kernel_fyz/include/ck ck_tile/include/
#     ln -sf $script_dir/../composable_kernel_fyz/include/ck_tile ck_tile/include/
# else
#     echo "Warning: Composable Kernel directory not found, creating placeholder directories"
#     mkdir -p ck_tile/include/ck
#     mkdir -p ck_tile/include/ck_tile
# fi

rm -rf build dist
rm -rf *.egg-info
python3 setup.py build_ext --inplace

# Find the .so file in build directory and create symlink in current directory
so_file=$(find build -name "*.so" -type f | head -n 1)
if [ -n "$so_file" ]; then
    ln -sf "$so_file" .
else
    echo "Error: No SO file found in build directory" >&2
    exit 1
fi

# Open users' original directory
cd "$original_dir"
