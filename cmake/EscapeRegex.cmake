# Define a function for escaping regex chars from strings
# This prevents regex failures when user provided paths contain regex special chars
function(escape_regex IN_STRING OUT_VAR)
    # https://gitlab.kitware.com/cmake/cmake/-/issues/18580#note_483128
    string(REGEX REPLACE "([][+.*()^])" "\\\\\\1" OUT_STRING "${IN_STRING}")
    set(${OUT_VAR} "${OUT_STRING}" PARENT_SCOPE)
endfunction()
