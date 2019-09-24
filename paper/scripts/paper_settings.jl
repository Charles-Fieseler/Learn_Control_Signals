# Settings to be used by the scripts in the 'paper' folder

FIGURE_FOLDERNAME = "C:\\Users\\charl\\Documents\\Current_work\\Learning_controllers\\PkgSRA\\paper\\figures\\"
EXAMPLE_FOLDERNAME = "C:\\Users\\charl\\Documents\\Current_work\\Learning_controllers\\PkgSRA\\examples\\"

# COLOR_DICT = Dict("true"=>[0, 0, 0],
#               "data"=>[111, 111, 111],
#               "intrinsic_true"=>[0, 0, 140],
#               "intrinsic"=>[0, 0, 227],
#               "control_true"=>[140, 0, 0],
#               "control_time"=>[227, 0, 0],
#               "control_space"=>[255, 128, 0])
COLOR_DICT = Dict("true"=>:black, # Should be dashed
              "data"=>:grey,
              "model_uncontrolled"=>:teal,
              "model_controlled"=>:deepskyblue,
              "prediction"=>:black,
              "intrinsic_true"=>:mediumblue,
              "intrinsic"=>:deepskyblue,
              "residual"=>:blue,
              "control_true"=>:red,
              "control_time"=>:darkorange,
              "control_space"=>:darkorange)
