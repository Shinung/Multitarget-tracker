# ----------------------------------------------------------------------------
# минимальная версия CMake, способная переварить этот файл
# ----------------------------------------------------------------------------
cmake_minimum_required(VERSION 2.8)
project(MultitargetTracker)

################################################################################################
# Short command for cuDNN detection. Believe it soon will be a part of CUDA toolkit distribution.
# That's why not FindcuDNN.cmake file, but just the macro
# Usage:
#   detect_cuDNN()
function(detect_cuDNN)
  set(CUDNN_ROOT "" CACHE PATH "CUDNN root folder")

  find_path(CUDNN_INCLUDE cudnn.h
            PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDA_TOOLKIT_INCLUDE}
            PATH_SUFFIXES include
            DOC "Path to cuDNN include directory." )
           
  unset(_path_suffixes)
  if(MSVC AND ${CMAKE_SIZEOF_VOID_P} EQUAL 8)
    set(_path_suffixes PATH_SUFFIXES lib/x64)
  else()
    set(_path_suffixes PATH_SUFFIXES lib/Win32)    
  endif()

  # dynamic libs have different suffix in mac and linux
  if(APPLE)
    set(CUDNN_LIB_NAME "libcudnn.dylib")
  elseif(MSVC)
    set(CUDNN_LIB_NAME "cudnn")
  else()
    set(CUDNN_LIB_NAME "libcudnn.so")
  endif()

  get_filename_component(__libpath_hist ${CUDA_CUDART_LIBRARY} PATH)
  find_library(CUDNN_LIBRARY NAMES ${CUDNN_LIB_NAME}
   PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDNN_INCLUDE} ${__libpath_hist} ${__libpath_hist}/../lib
   ${_path_suffixes}
   DOC "Path to cuDNN library.")
  
  if(CUDNN_INCLUDE AND CUDNN_LIBRARY)
    set(HAVE_CUDNN  TRUE PARENT_SCOPE)
    set(CUDNN_FOUND TRUE PARENT_SCOPE)

    file(READ ${CUDNN_INCLUDE}/cudnn.h CUDNN_VERSION_FILE_CONTENTS)

    # cuDNN v3 and beyond
    string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
           CUDNN_VERSION_MAJOR "${CUDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
           CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR}")
    string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
           CUDNN_VERSION_MINOR "${CUDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
           CUDNN_VERSION_MINOR "${CUDNN_VERSION_MINOR}")
    string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
           CUDNN_VERSION_PATCH "${CUDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
           CUDNN_VERSION_PATCH "${CUDNN_VERSION_PATCH}")

    if(NOT CUDNN_VERSION_MAJOR)
      set(CUDNN_VERSION "???")
    else()
      set(CUDNN_VERSION "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
    endif()

    message(STATUS "Found cuDNN: ver. ${CUDNN_VERSION} found (include: ${CUDNN_INCLUDE}, library: ${CUDNN_LIBRARY})")

    string(COMPARE LESS "${CUDNN_VERSION_MAJOR}" 3 cuDNNVersionIncompatible)
    if(cuDNNVersionIncompatible)
      message(FATAL_ERROR "cuDNN version >3 is required.")
    endif()

    set(CUDNN_VERSION "${CUDNN_VERSION}" PARENT_SCOPE)
    mark_as_advanced(CUDNN_INCLUDE CUDNN_LIBRARY CUDNN_ROOT)

  endif()
endfunction()

unset(CMAKE_CXX_FLAGS CACHE)

find_package(OpenMP)
if (OPENMP_FOUND)
    list(APPEND CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    list(APPEND CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic-errors -std=c++14" CACHE STRING COMPILE_FLAGS FORCE)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -march=native -mtune=native --fast-math -ffast-math -funroll-loops -Wall -DNDEBUG -DBOOST_DISABLE_ASSERTS" CACHE STRING COMPILE_FLAGS FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -march=native -mtune=native -Wall -DDEBUG" CACHE STRING COMPILE_FLAGS FORCE)
elseif (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /W4 -DGTL_STATIC" CACHE STRING COMPILE_FLAGS FORCE)
endif()


set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# ----------------------------------------------------------------------------
# Предполагаем, что FindOpenCV.cmake расположен по адресу CMAKE_MODULE_PATH.
# ----------------------------------------------------------------------------
  FIND_PACKAGE(OpenCV)
# ----------------------------------------------------------------------------
# ищем все .cpp и .h файлы и добавляем их в наш проект
# ----------------------------------------------------------------------------
  set(folder_source
             main.cpp
             VideoExample.cpp

             Detector/BaseDetector.cpp
             Detector/MotionDetector.cpp
             Detector/BackgroundSubtract.cpp
             Detector/vibe_src/vibe.cpp
             Detector/Subsense/BackgroundSubtractorLBSP.cpp
             Detector/Subsense/BackgroundSubtractorLOBSTER.cpp
             Detector/Subsense/BackgroundSubtractorSuBSENSE.cpp
             Detector/Subsense/LBSP.cpp
             Detector/FaceDetector.cpp
             Detector/PedestrianDetector.cpp
             Detector/pedestrians/c4-pedestrian-detector.cpp
             Detector/SSDMobileNetDetector.cpp
             Detector/YoloDetector.cpp
### Custom code ###########################################################
             Detector/SSDCustomNetDetector.cpp
             Detector/gpu_allocator.cpp
             Detector/SSD.cpp
###########################################################################
             Tracker/Ctracker.cpp
             Tracker/track.cpp
             Tracker/HungarianAlg/HungarianAlg.cpp
             Tracker/LocalTracker.cpp
             Tracker/Kalman.cpp
)

  set(folder_headers
             MouseExample.h
             VideoExample.h
             nms.h
             defines.h

             Detector/BaseDetector.h
             Detector/MotionDetector.h
             Detector/BackgroundSubtract.h
             Detector/vibe_src/vibe.hpp
             Detector/Subsense/BackgroundSubtractorLBSP.h
             Detector/Subsense/BackgroundSubtractorLOBSTER.h
             Detector/Subsense/BackgroundSubtractorSuBSENSE.h
             Detector/Subsense/DistanceUtils.h
             Detector/Subsense/LBSP.h
             Detector/Subsense/RandUtils.h
             Detector/FaceDetector.h
             Detector/PedestrianDetector.h
             Detector/pedestrians/c4-pedestrian-detector.h
             Detector/SSDMobileNetDetector.h
             Detector/YoloDetector.h
### Custom code ###########################################################
             Detector/SSDCustomNetDetector.h
             Detector/gpu_allocator.h
             Detector/SSD.h
             Detector/common.h
###########################################################################
             Tracker/Ctracker.h
             Tracker/track.h
             Tracker/HungarianAlg/HungarianAlg.h
             Tracker/LocalTracker.h
             Tracker/Kalman.h
)

  set(graph_source
             Tracker/graph/tokenise.cpp
#             Tracker/graph/script.cpp
#             Tracker/graph/rings.cpp
             Tracker/graph/mytree.cpp
             Tracker/graph/mygraph.cpp
             Tracker/graph/mwbmatching.cpp
             Tracker/graph/mincut.cpp
             Tracker/graph/gport.cpp
#             Tracker/graph/gml2nestedsql.cpp
#             Tracker/graph/gml2dot.cpp
#             Tracker/graph/components.cpp
             Tracker/graph/fheap.c
             )

  set(graph_header
             Tracker/graph/tokenise.h
             Tracker/graph/mytree.h
             Tracker/graph/mygraph.h
             Tracker/graph/mwbmatching.h
             Tracker/graph/mincut.h
             Tracker/graph/gport.h
             Tracker/graph/gdefs.h
             Tracker/graph/fheap.h
             )

  set(gtl_source
             Tracker/graph/GTL/src/bellman_ford.cpp
             Tracker/graph/GTL/src/bfs.cpp
             Tracker/graph/GTL/src/biconnectivity.cpp
             Tracker/graph/GTL/src/bid_dijkstra.cpp
             Tracker/graph/GTL/src/components.cpp
             Tracker/graph/GTL/src/debug.cpp
             Tracker/graph/GTL/src/dfs.cpp
             Tracker/graph/GTL/src/dijkstra.cpp
             Tracker/graph/GTL/src/edge.cpp
             Tracker/graph/GTL/src/embedding.cpp
             Tracker/graph/GTL/src/fm_partition.cpp
             Tracker/graph/GTL/src/gml_parser.cpp
             Tracker/graph/GTL/src/gml_scanner.cpp
             Tracker/graph/GTL/src/graph.cpp
             Tracker/graph/GTL/src/maxflow_ff.cpp
             Tracker/graph/GTL/src/maxflow_pp.cpp
             Tracker/graph/GTL/src/maxflow_sap.cpp
             Tracker/graph/GTL/src/min_tree.cpp
             Tracker/graph/GTL/src/node.cpp
             Tracker/graph/GTL/src/planarity.cpp
             Tracker/graph/GTL/src/pq_node.cpp
             Tracker/graph/GTL/src/pq_tree.cpp
             Tracker/graph/GTL/src/ratio_cut_partition.cpp
             Tracker/graph/GTL/src/st_number.cpp
             Tracker/graph/GTL/src/topsort.cpp
             )

  set(gtl_header
             Tracker/graph/GTL/include/GTL/algorithm.h
             Tracker/graph/GTL/include/GTL/bellman_ford.h
             Tracker/graph/GTL/include/GTL/bfs.h
             Tracker/graph/GTL/include/GTL/biconnectivity.h
             Tracker/graph/GTL/include/GTL/bid_dijkstra.h
             Tracker/graph/GTL/include/GTL/bin_heap.h
             Tracker/graph/GTL/include/GTL/components.h
             Tracker/graph/GTL/include/GTL/debug.h
             Tracker/graph/GTL/include/GTL/dfs.h
             Tracker/graph/GTL/include/GTL/dijkstra.h
             Tracker/graph/GTL/include/GTL/edge.h
             Tracker/graph/GTL/include/GTL/edge_data.h
             Tracker/graph/GTL/include/GTL/edge_map.h
             Tracker/graph/GTL/include/GTL/embedding.h
             Tracker/graph/GTL/include/GTL/fm_partition.h
             Tracker/graph/GTL/include/GTL/gml_parser.h
             Tracker/graph/GTL/include/GTL/gml_scanner.h
             Tracker/graph/GTL/include/GTL/graph.h
             Tracker/graph/GTL/include/GTL/maxflow_ff.h
             Tracker/graph/GTL/include/GTL/maxflow_pp.h
             Tracker/graph/GTL/include/GTL/maxflow_sap.h
             Tracker/graph/GTL/include/GTL/min_tree.h
             Tracker/graph/GTL/include/GTL/ne_map.h
             Tracker/graph/GTL/include/GTL/node.h
             Tracker/graph/GTL/include/GTL/node_data.h
             Tracker/graph/GTL/include/GTL/node_map.h
             Tracker/graph/GTL/include/GTL/planarity.h
             Tracker/graph/GTL/include/GTL/pq_node.h
             Tracker/graph/GTL/include/GTL/pq_tree.h
             Tracker/graph/GTL/include/GTL/ratio_cut_partition.h
             Tracker/graph/GTL/include/GTL/st_number.h
             Tracker/graph/GTL/include/GTL/symlist.h
             Tracker/graph/GTL/include/GTL/topsort.h
             Tracker/graph/GTL/include/GTL/version.h
             Tracker/graph/GTL/include/GTL/GTL.h
             )

  SOURCE_GROUP("Source Files" FILES ${folder_source})
  SOURCE_GROUP("Header Files" FILES ${folder_headers})

  SOURCE_GROUP("graph" FILES ${graph_source} ${graph_header})
  SOURCE_GROUP("GTL" FILES ${gtl_source} ${gtl_header})

include(CheckIncludeFileCXX)
check_include_file_cxx(opencv2/bgsegm.hpp HAVE_OPENCV_CONTRIB)
if(HAVE_OPENCV_CONTRIB)
    add_definitions(-DHAVE_OPENCV_CONTRIB)

    option(USE_OCV_BGFG "Should use the bgfg algorithms from opencv_contrib?" ON)
    option(USE_OCV_KCF "Should use the KCF tracker from opencv_contrib for collisions resolving?" ON)
    option(USE_OCV_UKF "Should use the Unscented Kalman Filterr from opencv_contrib?" ON)

else(HAVE_OPENCV_CONTRIB)

    option(USE_OCV_BGFG "Should use the bgfg algorithms from opencv_contrib?" OFF)
    option(USE_OCV_KCF "Should use the KCF tracker from opencv_contrib for collisions resolving?" OFF)
    option(USE_OCV_UKF "Should use the Unscented Kalman Filterr from opencv_contrib?" OFF)
endif(HAVE_OPENCV_CONTRIB)

if(USE_OCV_UKF)
    add_definitions(-DUSE_OCV_UKF)
endif(USE_OCV_UKF)

if(USE_OCV_KCF)
    add_definitions(-DUSE_OCV_KCF)
endif(USE_OCV_KCF)

if(USE_OCV_BGFG)
    add_definitions(-DUSE_OCV_BGFG)
endif(USE_OCV_BGFG)

option(USE_CAFFE "Use Specify Caffe library for Custom trained .caffemodel" OFF)

if(USE_CAFFE)
    find_package(CUDA)
    detect_cuDNN()

    add_definitions(-DCAFFE_VERSION=1.0.0-rc3)
    add_definitions(-DBOOST_ALL_NO_LIB)
    add_definitions(-DUSE_CUDNN)
    add_definitions(-DUSE_OPENCV)
    add_definitions(-DOPENCV_VERSION=3)
    add_definitions(-DCMAKE_WINDOWS_BUILD)
    add_definitions(-DGLOG_NO_ABBREVIATED_SEVERITIES)
    add_definitions("-DGOOGLE_GLOG_DLL_DECL=__declspec(dllimport)")
    add_definitions("-DGOOGLE_GLOG_DLL_DECL_FOR_UNITTESTS=__declspec(dllimport)")

    set(CAFFE_INCLUDE_DIR "" CACHE PATH "Caffe include directory")
    set(CAFFE_LIBS_DIR "" CACHE PATH "Caffe library directory")
    set(CAFFE_DEPENDENCIES_INCLUDE_DIR "" CACHE PATH "Caffe's dependencies include directory")
    set(CAFFE_DEPENDENCIES_LIBS_DIR "" CACHE PATH "Caffe's dependencies directory")
    set(PYTHON_LIBS_DIR "" CACHE PATH "Python library directory")

    INCLUDE_DIRECTORIES(${CAFFE_INCLUDE_DIR})
    INCLUDE_DIRECTORIES(${CAFFE_DEPENDENCIES_INCLUDE_DIR})
    INCLUDE_DIRECTORIES(${CAFFE_DEPENDENCIES_INCLUDE_DIR}/boost-1_61)
    INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
    INCLUDE_DIRECTORIES(${CUDNN_INCLUDE})
endif(USE_CAFFE)

# ----------------------------------------------------------------------------  
# создаем проект
# ----------------------------------------------------------------------------
  ADD_EXECUTABLE(MultitargetTracker ${folder_source} ${folder_headers} ${graph_source} ${graph_header} ${gtl_source} ${gtl_header})
# ----------------------------------------------------------------------------
# добавляем include директории
# ----------------------------------------------------------------------------
  INCLUDE_DIRECTORIES(
		      ${PROJECT_SOURCE_DIR}/Detector
                      ${PROJECT_SOURCE_DIR}/Detector/vibe_src
                      ${PROJECT_SOURCE_DIR}/Detector/Subsense
                      ${PROJECT_SOURCE_DIR}/Tracker
                      ${PROJECT_SOURCE_DIR}/Tracker/HungarianAlg
                      ${PROJECT_SOURCE_DIR}/Tracker/graph
                      ${PROJECT_SOURCE_DIR}/Tracker/graph/GTL/include)

  INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})
  INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

# ----------------------------------------------------------------------------
# и Lib-ы opencv
# ----------------------------------------------------------------------------


if (CMAKE_COMPILER_IS_GNUCXX)
    set(LIBS
        ${OpenCV_LIBS}
    #    iconv
    )
else(CMAKE_COMPILER_IS_GNUCXX)
    if(USE_CAFFE)
        set(LIBS
            debug ${CAFFE_LIBS_DIR}/caffe-d.lib optimized ${CAFFE_LIBS_DIR}/caffe.lib
            debug ${CAFFE_LIBS_DIR}/caffeproto-d.lib optimized ${CAFFE_LIBS_DIR}/caffeproto.lib
            debug ${CAFFE_DEPENDENCIES_LIBS_DIR}/boost_system-vc140-mt-gd-1_61.lib optimized ${CAFFE_DEPENDENCIES_LIBS_DIR}/boost_system-vc140-mt-1_61.lib
            debug ${CAFFE_DEPENDENCIES_LIBS_DIR}/boost_thread-vc140-mt-gd-1_61.lib optimized ${CAFFE_DEPENDENCIES_LIBS_DIR}/boost_thread-vc140-mt-1_61.lib
            debug ${CAFFE_DEPENDENCIES_LIBS_DIR}/boost_filesystem-vc140-mt-gd-1_61.lib optimized ${CAFFE_DEPENDENCIES_LIBS_DIR}/boost_filesystem-vc140-mt-1_61.lib
            debug ${CAFFE_DEPENDENCIES_LIBS_DIR}/boost_chrono-vc140-mt-gd-1_61.lib optimized ${CAFFE_DEPENDENCIES_LIBS_DIR}/boost_date_time-vc140-mt-1_61.lib
            debug ${CAFFE_DEPENDENCIES_LIBS_DIR}/boost_date_time-vc140-mt-gd-1_61.lib optimized ${CAFFE_DEPENDENCIES_LIBS_DIR}/boost_chrono-vc140-mt-1_61.lib
            debug ${CAFFE_DEPENDENCIES_LIBS_DIR}/boost_atomic-vc140-mt-gd-1_61.lib optimized ${CAFFE_DEPENDENCIES_LIBS_DIR}/boost_atomic-vc140-mt-1_61.lib
            debug ${CAFFE_DEPENDENCIES_LIBS_DIR}/boost_python-vc140-mt-gd-1_61.lib optimized ${CAFFE_DEPENDENCIES_LIBS_DIR}/boost_python-vc140-mt-1_61.lib
            ${PYTHON_LIBS_DIR}/python35.lib
            debug ${CAFFE_DEPENDENCIES_LIBS_DIR}/glogd.lib optimized ${CAFFE_DEPENDENCIES_LIBS_DIR}/glog.lib
            debug ${CAFFE_DEPENDENCIES_LIBS_DIR}/gflagsd.lib optimized ${CAFFE_DEPENDENCIES_LIBS_DIR}/gflags.lib
            debug ${CAFFE_DEPENDENCIES_LIBS_DIR}/libprotobufd.lib optimized ${CAFFE_DEPENDENCIES_LIBS_DIR}/libprotobuf.lib
            debug ${CAFFE_DEPENDENCIES_LIBS_DIR}/caffehdf5_hl_D.lib optimized ${CAFFE_DEPENDENCIES_LIBS_DIR}/caffehdf5_hl.lib
            debug ${CAFFE_DEPENDENCIES_LIBS_DIR}/caffehdf5_D.lib optimized ${CAFFE_DEPENDENCIES_LIBS_DIR}/caffehdf5.lib
            debug ${CAFFE_DEPENDENCIES_LIBS_DIR}/caffezlibd.lib optimized ${CAFFE_DEPENDENCIES_LIBS_DIR}/caffezlib.lib
            debug ${CAFFE_DEPENDENCIES_LIBS_DIR}/lmdbd.lib optimized ${CAFFE_DEPENDENCIES_LIBS_DIR}/lmdb.lib
            debug ${CAFFE_DEPENDENCIES_LIBS_DIR}/leveldbd.lib optimized ${CAFFE_DEPENDENCIES_LIBS_DIR}/leveldb.lib
            debug ${CAFFE_DEPENDENCIES_LIBS_DIR}/snappy_staticd.lib optimized ${CAFFE_DEPENDENCIES_LIBS_DIR}/snappy_static.lib
            ${CAFFE_DEPENDENCIES_LIBS_DIR}/libopenblas.dll.a
            ${CUDA_CUDART_LIBRARY}
            ${CUDA_curand_LIBRARY}
            ${CUDA_CUBLAS_LIBRARIES}
            ${CUDNN_LIBRARY}
            ${OpenCV_LIBS}
            ntdll.lib
        )
    else(USE_CAFFE)
        set(LIBS
            ${OpenCV_LIBS}
        )
    endif()
endif()
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${LIBS})
