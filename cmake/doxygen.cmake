# Doxygen
find_package(Doxygen OPTIONAL_COMPONENTS mscgen dia dot)
if(DOXYGEN_FOUND)
	option(BUILD_API_DOCUMENTATION "Enable building documentation (requires Doxygen)" ON)
else()
	if(CMAKE_CUDA_COMPILER STREQUAL NOTFOUND)
		message(FATAL_ERROR 
			" Doxygen: NOT FOUND!\n"
			" Documentation project cannot be generated.\n"
			" Please install Doxygen and re-run configure.")
	
	else()
		message( 
			" Doxygen: NOT FOUND!\n"
			" Documentation project cannot be generated.\n"
			" Please install Doxygen and re-run configure if required.")
	endif()
endif()

macro(create_pydoxygen_target FLAMEGPU_ROOT DOXY_OUT_DIR XML_PATH)
        # Py docs, needs SWIG to convert the .i file/s to .py
        # We don't require compilation of this file, as performed by UseSWIG module, so we will create a seperate target
        # So we create a minimal SWIG target ourselves pyflamegpu.py ourselves
        if(NOT SWIG_FOUND)
            find_package(SWIG ${SWIG_MINIMUM_SUPPORTED_VERSION})
            # Borrowed from swig/CMakeLists.txt, should really move it to a module or something.
            # If minimum required swig could not be found, download and build it.
            if(NOT SWIG_FOUND)
                # SWIG_DIR and SWIG_EXECUTABLE are used by FindSwig. Executable specifies version info etc, swig_dir should contain swig.swg (i.e the value of swig -swiglib so doesn't need specifying if teh swig install is good?)

                # Unset swig related variables from the previous failed attempt at finding swig
                unset(SWIG_FOUND)
                unset(SWIG_VERSION)
                unset(SWIG_DIR)
                unset(SWIG_EXECUTABLE)
                unset(SWIG_FOUND CACHE)
                unset(SWIG_VERSION CACHE)
                unset(SWIG_DIR CACHE)
                unset(SWIG_EXECUTABLE CACHE)


                if(WIN32)
                    # Download pre-compiled swig on windows.
                    FetchContent_Declare(
                        swig
                        URL      http://prdownloads.sourceforge.net/swig/swigwin-${SWIG_DOWNLOAD_VERSION}.zip
                        # URL_HASH #@todo - make sure the hash of the download is good?
                    )
                    FetchContent_GetProperties(swig)
                    if(NOT swig_POPULATED)
                        message(STATUS "[swig] Downloading swigwin-${SWIG_DOWNLOAD_VERSION}.zip")
                        # Download
                        FetchContent_Populate(swig)
                        
                        # Set variables used by find_swig to find swig.
                        # Set locally and in the cache for subsequent invocations of CMake
                        set(SWIG_EXECUTABLE "${swig_SOURCE_DIR}/swig.exe")
                        set(SWIG_EXECUTABLE "${swig_SOURCE_DIR}/swig.exe" CACHE FILEPATH "Path to SWIG executable")
                    endif()
                else()
                    # Under linux, download the .tar.gz, extract, build and install.
                    # This must be done at configure time, as FindSwig requires the swig executable.
                    # FetchContent allows download at configure time, but must use execute_process to run commands at configure time.

                    # Download from sourceforge not github, github releases require several extra tools to build (bison, yacc, autoconfigure, automake), and we want to keep dependencies light. 
                    FetchContent_Declare(
                        swig
                        URL      https://downloads.sourceforge.net/project/swig/swig/swig-${SWIG_DOWNLOAD_VERSION}/swig-${SWIG_DOWNLOAD_VERSION}.tar.gz
                        # URL_HASH #@todo - make sure the hash of the download is good?
                    )
                    FetchContent_GetProperties(swig)
                    if(NOT swig_POPULATED)
                        message(STATUS "[swig] Downloading swig-${SWIG_DOWNLOAD_VERSION}.tar.gz")
                        # Download the content
                        FetchContent_Populate(swig)

                        # Set some paths for error files in case things go wrong.
                        set(swig_configure_ERROR_FILE "${swig_BINARY_DIR}/swig-error-configue.log")
                        set(swig_make_ERROR_FILE "${swig_BINARY_DIR}/swig-error-make.log")
                        set(swig_makeinstall_ERROR_FILE "${swig_BINARY_DIR}/swig-error-make-install.log")

                        # run ./configure with an appropraite prefix to install into the _deps/swig-bin directory
                        message(STATUS "[swig] ./configure --prefix ${swig_BINARY_DIR}")
                        execute_process(
                            COMMAND "./configure" "--prefix" "${swig_BINARY_DIR}"
                            WORKING_DIRECTORY ${swig_SOURCE_DIR}
                            RESULT_VARIABLE swig_configure_RESULT
                            OUTPUT_VARIABLE swig_configure_OUTPUT
                            ERROR_FILE ${swig_configure_ERROR_FILE}
                        )
                        if(NOT swig_configure_RESULT EQUAL "0")
                            message(FATAL_ERROR 
                            " [swig] Error configuring SWIG ${SWIG_DOWNLOAD_VERSION}.\n"
                            " Error log: ${swig_configure_ERROR_FILE}\n"
                            " Consider installing SWIG ${SWIG_MINIMUM_SUPPORTED_VERSION} yourself and passing -DSWIG_EXECUTABLE=/path/to/swig.")
                        endif()

                        # run make to compile swig within swig_SOURCE_DIR
                        message(STATUS "[swig] make")
                        execute_process(
                            COMMAND "make"
                            WORKING_DIRECTORY ${swig_SOURCE_DIR}
                            RESULT_VARIABLE swig_make_RESULT
                            OUTPUT_VARIABLE swig_make_OUTPUT
                        )
                        if(NOT swig_make_RESULT EQUAL "0")
                            message(FATAL_ERROR 
                            " [swig] Error compiling SWIG ${SWIG_DOWNLOAD_VERSION}\n"
                            " Error log: ${swig_make_ERROR_FILE}"
                            " Consider installing SWIG >= ${SWIG_MINIMUM_SUPPORTED_VERSION} yourself and passing -DSWIG_EXECUTABLE=/path/to/swig.")
                        endif()

                        # run make install to copy to the installation location (swig_BINARY_DIR)
                        message(STATUS "[swig] make install")
                        execute_process(
                            COMMAND "make" "install"
                            WORKING_DIRECTORY ${swig_SOURCE_DIR}
                            RESULT_VARIABLE swig_makeinstall_RESULT
                            OUTPUT_VARIABLE swig_makeinstall_OUTPUT
                        )
                        if(NOT swig_makeinstall_RESULT EQUAL "0")
                            message(FATAL_ERROR 
                            " [swig] Error installing SWIG ${SWIG_DOWNLOAD_VERSION}\n"
                            " Error log: ${swig_makeinstall_ERROR_FILE}"
                            " Consider installing SWIG >= ${SWIG_MINIMUM_SUPPORTED_VERSION} yourself and passing -DSWIG_EXECUTABLE=/path/to/swig.")
                        endif()

                        # Set variables used by find_swig to find swig.
                        # Set locally and in the cache for subsequent invocations of CMake
                        set(SWIG_EXECUTABLE "${swig_BINARY_DIR}/bin/swig")
                        set(SWIG_EXECUTABLE "${swig_BINARY_DIR}/bin/swig" CACHE FILEPATH "Path to SWIG executable")
                    endif()
                endif()
                # Attempt to find swig again, but as REQUIRED.
                find_package(SWIG ${SWIG_MINIMUM_SUPPORTED_VERSION} REQUIRED)
            endif()
        endif()
#"C:\Program Files\CMake\bin\cmake.exe" -E make_directory C:/Users/Robadob/fgpu2/build/swig/python//pyflamegpu C:/Users/Robadob/fgpu2/build/swig/python//pyflamegpu
#if %errorlevel% neq 0 goto :cmEnd
#"C:\Program Files\CMake\bin\cmake.exe" -E env SWIG_LIB=C:/Users/Robadob/fgpu2/build/_deps/swig-src/Lib C:/Users/Robadob/fgpu2/build/_deps/swig-src/swig.exe -python -doxygen -IC:/Users/Robadob/fgpu2/include -IC:/ProgramData/Anaconda3/include -IC:/Users/Robadob/fgpu2_vis/include "-IC:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/include" -IC:/Users/Robadob/fgpu2/build/_deps/jitify-src -IC:/Users/Robadob/fgpu2/include -IC:/Users/Robadob/fgpu2/build/FLAMEGPU2/_deps/flamegpu2_visualiser-build/freetype-build/include -IC:/Users/Robadob/fgpu2/build/FLAMEGPU2/_deps/flamegpu2_visualiser-build/freetype-src/include -IC:/Users/Robadob/fgpu2/build/_cmrc/include -IC:/Users/Robadob/fgpu2/build/_deps/thrust-src/dependencies/cub -IC:/Users/Robadob/fgpu2/build/_deps/thrust-src -IC:/Users/Robadob/fgpu2/build/_deps/tinyxml2-src -DVISUALISATION -DSEATBELTS=1 -DNOMINMAX -threads -outdir C:/Users/Robadob/fgpu2/build/swig/python//pyflamegpu -c++ -module pyflamegpu -interface _pyflamegpu -o C:/Users/Robadob/fgpu2/build/swig/python//pyflamegpu/flamegpuPYTHON_wrap.cxx C:/Users/Robadob/fgpu2/swig/python/flamegpu.i
        if(SEATBELTS)
          set(DEFINE_SEATBELTS "-DSEATBELTS=1")
        endif()
        if(VISUALISATION)
          set(DEFINE_VISUALISATION "-I${VISUALISATION_ROOT}/include -DVISUALISATION")
        endif()
        set(TRANSLATE_DOXYGEN "-doxygen")
        add_custom_target(
            pyflamegpu_pyonly
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/pyflamegpu_docs"
            COMMAND ${CMAKE_COMMAND} -E env "SWIG_LIB=${SWIG_DIR}" ${SWIG_EXECUTABLE} -python ${TRANSLATE_DOXYGEN} "-I${FLAMEGPU_ROOT}/include" ${DEFINE_VISUALISATION} ${DEFINE_SEATBELTS} -DNOMINMAX -threads -outdir "${CMAKE_BINARY_DIR}/pyflamegpu_docs" -c++ -module pyflamegpu -interface _pyflamegpu -o "${CMAKE_BINARY_DIR}/pyflamegpu_docs/flamegpuPYTHON_wrap.cxx" "${FLAMEGPU_ROOT}/swig/python/flamegpu.i"
        )
        # And now do the doxygen bits
        set(DOXYGEN_STRIP_FROM_PATH       "${CMAKE_BINARY_DIR}/pyflamegpu_docs")
        set(DOXYGEN_STRIP_FROM_INC_PATH   "${CMAKE_BINARY_DIR}/pyflamegpu_docs")
        set(DOXYGEN_PREDEFINED            "")
        set(DOXYGEN_PYTHON_DOCSTRING      NO)
        set(DOXY_INPUT_FILES "${CMAKE_BINARY_DIR}/pyflamegpu_docs/pyflamegpu.py")
        set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/pyflamegpu_docs")
        if("${XML_PATH}" STREQUAL "")
            doxygen_add_docs("pydocs" "${DOXY_INPUT_FILES}")
            set_target_properties("pydocs" PROPERTIES EXCLUDE_FROM_ALL TRUE)
            if(COMMAND CMAKE_SET_TARGET_FOLDER)
                # Put within FLAMEGPU filter
                CMAKE_SET_TARGET_FOLDER("pydocs" "FLAMEGPU")
            endif()
            add_dependencies(pydocs pyflamegpu_pyonly)
        else()
            doxygen_add_docs("api_pydocs_xml" "${DOXY_INPUT_FILES}")
            set_target_properties("api_pydocs_xml" PROPERTIES EXCLUDE_FROM_ALL TRUE)
            if(COMMAND CMAKE_SET_TARGET_FOLDER)
                # Put within FLAMEGPU filter
                CMAKE_SET_TARGET_FOLDER("api_pydocs_xml" "FLAMEGPU")
            endif()
            add_dependencies(api_pydocs_xml pyflamegpu_pyonly)
        endif()
endmacro()

function(create_doxygen_target FLAMEGPU_ROOT DOXY_OUT_DIR XML_PATH)
    if(DOXYGEN_FOUND)
        # Modern method which generates unique doxyfile
        # These args taken from readme.md at time of commit
        set(DOXYGEN_OUTPUT_DIRECTORY "${DOXY_OUT_DIR}")
        set(DOXYGEN_PROJECT_NAME "FLAMEGPU 2.0")
        set(DOXYGEN_PROJECT_NUMBER "")
        set(DOXYGEN_PROJECT_BRIEF "Expansion of FLAMEGPU to provide middle-ware for complex systems simulations to utilise CUDA.")
        set(DOXYGEN_GENERATE_LATEX        NO)
        set(DOXYGEN_EXTRACT_ALL           YES)
        set(DOXYGEN_CLASS_DIAGRAMS        YES)
        set(DOXYGEN_HIDE_UNDOC_RELATIONS  NO)
        set(DOXYGEN_CLASS_GRAPH           YES)
        set(DOXYGEN_COLLABORATION_GRAPH   YES)
        set(DOXYGEN_UML_LOOK              YES)
        set(DOXYGEN_UML_LIMIT_NUM_FIELDS  50)
        set(DOXYGEN_TEMPLATE_RELATIONS    YES)
        set(DOXYGEN_DOT_TRANSPARENT       NO)
        set(DOXYGEN_CALL_GRAPH            YES)
        set(DOXYGEN_RECURSIVE             YES)
        set(DOXYGEN_CALLER_GRAPH          YES)
        set(DOXYGEN_GENERATE_TREEVIEW     YES)
        set(DOXYGEN_EXTRACT_PRIVATE       YES)
        set(DOXYGEN_EXTRACT_STATIC        YES)
        set(DOXYGEN_EXTRACT_LOCAL_METHODS NO)
        set(DOXYGEN_FILE_PATTERNS         "*.h" "*.cuh" "*.c" "*.cpp" "*.cu" "*.cuhpp" "*.md" "*.hh" "*.hxx" "*.hpp" "*.h++" "*.cc" "*.cxx" "*.c++")
        set(DOXYGEN_EXTENSION_MAPPING     "cu=C++" "cuh=C++" "cuhpp=C++")
        # Limit diagram graph node count / depth for simply diagrams.
        set(DOXYGEN_DOT_GRAPH_MAX_NODES   100)
        set(DOXYGEN_MAX_DOT_GRAPH_DEPTH   1)
        # Select diagram output format i.e png or svg
        set(DOXYGEN_DOT_IMAGE_FORMAT      png)
        # If using svg the interactivity can be enabled if desired.
        set(DOXYGEN_INTERACTIVE_SVG       NO)
        # Replace full absolute paths with relative paths to the project root.
        set(DOXYGEN_FULL_PATH_NAMES       YES)
        set(DOXYGEN_STRIP_FROM_PATH       ${FLAMEGPU_ROOT})
        set(DOXYGEN_STRIP_FROM_INC_PATH   ${FLAMEGPU_ROOT})
        # Upgrade warnings
        set(DOXYGEN_QUIET                 YES) # Supress non warning messages
        set(DOXYGEN_WARNINGS              YES)
        set(DOXYGEN_WARN_IF_UNDOCUMENTED  YES)
        set(DOXYGEN_WARN_IF_DOC_ERROR     YES)
        set(DOXYGEN_WARN_IF_INCOMPLETE_DOC YES)
        set(DOXYGEN_WARN_NO_PARAMDOC      YES) # Defaults off, unlike other warning settings
        if(WARNINGS_AS_ERRORS)
            if(DOXYGEN_VERSION VERSION_GREATER_EQUAL 1.9.0)
                set(DOXYGEN_WARN_AS_ERROR     FAIL_ON_WARNINGS)
            else()
                set(DOXYGEN_WARN_AS_ERROR     YES)
            endif()
        endif()
        # These are required for expanding FGPUException definition macros to be documented
        set(DOXYGEN_ENABLE_PREPROCESSING  YES)
        set(DOXYGEN_MACRO_EXPANSION       YES)
        set(DOXYGEN_EXPAND_ONLY_PREDEF    YES)
        set(DOXYGEN_PREDEFINED            "DERIVED_FGPUException(name,default_msg)=class name: public FGPUException { public: explicit name(const char *format = default_msg)\; }" "VISUALISATION= ")
        set(DOXY_INPUT_FILES              "${FLAMEGPU_ROOT}/include;${FLAMEGPU_ROOT}/src;${FLAMEGPU_ROOT}/README.md")
        # Create doxygen target            
        if("${XML_PATH}" STREQUAL "")
            set(DOXYGEN_GENERATE_HTML     YES)
            set(DOXYGEN_GENERATE_XML      NO)
            set(DOXYGEN_HTML_OUTPUT       docs)
            doxygen_add_docs("docs" "${DOXY_INPUT_FILES}")
            set_target_properties("docs" PROPERTIES EXCLUDE_FROM_ALL TRUE)
            if(COMMAND CMAKE_SET_TARGET_FOLDER)
                # Put within FLAMEGPU filter
                CMAKE_SET_TARGET_FOLDER("docs" "FLAMEGPU")
            endif()
        else()
            set(DOXYGEN_GENERATE_HTML     NO)
            set(DOXYGEN_GENERATE_XML      YES)
            set(DOXYGEN_XML_OUTPUT        "${XML_PATH}")
            doxygen_add_docs("api_docs_xml" "${DOXY_INPUT_FILES}")
            set_target_properties("api_docs_xml" PROPERTIES EXCLUDE_FROM_ALL TRUE)
            if(COMMAND CMAKE_SET_TARGET_FOLDER)
                # Put within FLAMEGPU filter
                CMAKE_SET_TARGET_FOLDER("api_docs_xml" "FLAMEGPU")
            endif()
        endif()
        create_pydoxygen_target("${FLAMEGPU_ROOT}" "${DOXY_OUT_DIR}" "${XML_PATH}")
    endif()  
endfunction()
