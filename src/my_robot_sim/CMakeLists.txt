install(
  PROGRAMS
    launch/dyn2d_policy_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)
