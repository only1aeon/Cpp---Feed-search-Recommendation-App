cmake_minimum_required(VERSION 3.15)
project(recommender_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

# Find FAISS - user must install FAISS and set FAISS_ROOT or ensure pkg-config/cmake can find it
find_library(FAISS_LIB faiss)
if(NOT FAISS_LIB)
  message(WARNING "FAISS library not found by find_library. You must install FAISS and set FAISS_LIB to the library path.")
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
file(GLOB SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")

add_executable(recommender_demo ${SOURCES})

if(FAISS_LIB)
  target_link_libraries(recommender_demo PRIVATE ${FAISS_LIB} Threads::Threads)
else()
  message(STATUS "Building without FAISS link - demo will compile but FAISS calls must be stubbed or library path set.")
endif()
