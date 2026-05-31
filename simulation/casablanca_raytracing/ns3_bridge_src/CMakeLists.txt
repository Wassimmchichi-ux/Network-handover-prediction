find_library(LIBZMQ_RUNTIME NAMES libzmq.so.5 zmq PATHS /lib64 /usr/lib64)

if(NOT LIBZMQ_RUNTIME)
  message(FATAL_ERROR "Unable to locate libzmq runtime library")
endif()

set(bridge_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/ns3-hybrid-mobility-server.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/hybrid-mobility-model.cc
)

build_exec(
  EXECNAME ns3-hybrid-mobility-server
  EXECNAME_PREFIX scratch_hybrid-nsa-zmq_
  SOURCE_FILES "${bridge_sources}"
  LIBRARIES_TO_LINK "${ns3-libs};${LIBZMQ_RUNTIME}"
  EXECUTABLE_DIRECTORY_PATH ${CMAKE_OUTPUT_DIRECTORY}/scratch/hybrid-nsa-zmq
)
