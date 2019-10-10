# Settings to be used by the scripts in the 'paper' folder

BASE_FOLDERNAME = "C:\\Users\\charl\\Documents\\Current_work\\Learning_controllers\\PkgSRA\\"
FIGURE_FOLDERNAME = BASE_FOLDERNAME*"paper\\figures\\"
EXAMPLE_FOLDERNAME = BASE_FOLDERNAME*"examples\\"
DAT_FOLDERNAME = BASE_FOLDERNAME*"paper\\dat\\"


# COLOR_DICT = Dict("true"=>[0, 0, 0],
#               "data"=>[111, 111, 111],
#               "intrinsic_true"=>[0, 0, 140],
#               "intrinsic"=>[0, 0, 227],
#               "control_true"=>[140, 0, 0],
#               "control_time"=>[227, 0, 0],
#               "control_space"=>[255, 128, 0])
# COLOR_DICT = Dict("true"=>:black, # Should be dashed
#               "data"=>:grey,
#               "model_uncontrolled"=>:teal,
#               "model_controlled"=>:deepskyblue,
#               "prediction"=>:black,
#               "intrinsic_true"=>:mediumblue,
#               "intrinsic"=>:deepskyblue,
#               "residual"=>:blue,
#               "control_true"=>:red,
#               "control_time"=>:darkorange,
#               "control_space"=>:darkorange)

  COLOR_DICT = Dict("true"=>:black, # Should be dashed
                "data"=>:grey,
                "data_uncontrolled"=>:grey,
                "model_uncontrolled"=>:blue,
                "model_partial"=>:teal,
                "model_controlled"=>:purple,
                "prediction"=>:black,
                "intrinsic_true"=>:blue,
                "intrinsic"=>:blue,
                "residual"=>:blue,
                "control_true"=>:black,
                "control_time"=>:red,
                "control_space"=>:red)
