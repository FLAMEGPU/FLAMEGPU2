include_guard(GLOBAL)

#-----------------------------------------------------------------------
# a macro that only sets the FOLDER target property if it's
# "appropriate"
# Borrowed from cmake's own CMakeLists.txt
#-----------------------------------------------------------------------
macro(flamegpu_set_target_folder tgt folder)
  if(CMAKE_USE_FOLDERS)
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
    if(TARGET ${tgt}) # AND MSVC # AND MSVC stops all lint from being set with folder
      set_property(TARGET "${tgt}" PROPERTY FOLDER "${folder}")
    endif()
  else()
    set_property(GLOBAL PROPERTY USE_FOLDERS OFF)
  endif()
endmacro()