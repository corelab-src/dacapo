
add_library (SEAL_HEVM
  SHARED
  SEAL_HEVM.cpp)
target_link_libraries(SEAL_HEVM
  PUBLIC
  SEAL::seal
  )

add_library (HEAAN_HEVM
  SHARED
  HEAAN_HEVM.cpp)
target_link_libraries(HEAAN_HEVM
  PUBLIC
  HEaaN::HEaaN
  )
