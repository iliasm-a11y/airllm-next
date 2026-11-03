# CMAKE_PROJECT_INCLUDE runs after project() at the root, before any target.
# OBJCXX must be enabled in the highest directory common to its link users.
if(APPLE AND GGML_CPU_METAL_Q8_BRIDGE)
    # Keep legacy .m files Objective-C when OBJCXX adds its .M mapping on macOS.
    enable_language(OBJCXX OBJC)
endif()
