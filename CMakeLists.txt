cmake_minimum_required(VERSION 3.10)
project(SMPLFitting)

set(CMAKE_CXX_STANDARD 14)

# ================= 1. 【新增】解决 Unicode 报错的关键 =================
add_compile_options(/utf-8)

# ================= 2. 你的库路径 (保持你原本正确的路径) =================
# 注意：请确保这些路径是你电脑上存在的！
set(Torch_DIR "D:/libtorch/libtorch-win-shared-with-deps-2.9.1+cu126/libtorch/share/cmake/Torch")
list(APPEND CMAKE_PREFIX_PATH "D:/Desktop/open3d-devel-windows-amd64-0.19.0/CMake")

find_package(Torch REQUIRED)
find_package(Open3D REQUIRED)

add_executable(smpl_fitting main.cpp)
target_link_libraries(smpl_fitting ${TORCH_LIBRARIES} Open3D::Open3D)

if (MSVC)
    add_custom_command(TARGET smpl_fitting POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TORCH_INSTALL_PREFIX}/lib/torch.dll" "$<TARGET_FILE_DIR:smpl_fitting>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TORCH_INSTALL_PREFIX}/lib/torch_cpu.dll" "$<TARGET_FILE_DIR:smpl_fitting>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TORCH_INSTALL_PREFIX}/lib/c10.dll" "$<TARGET_FILE_DIR:smpl_fitting>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TORCH_INSTALL_PREFIX}/lib/torch_cuda.dll" "$<TARGET_FILE_DIR:smpl_fitting>"
        # 如果 Open3D.dll 没自动过来，这里可以手动加一行复制 Open3D.dll
    )
endif()