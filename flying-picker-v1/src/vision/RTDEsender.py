import math
import time
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config

class Sender:
    def __init__(self, robot_ip, robot_port, config_file):
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.config_file = config_file
        self.con = None
        self.inputs = None

    def connect(self):
        conf = rtde_config.ConfigFile(self.config_file)

        output_names, output_types = conf.get_recipe("outputs")
        input_names, input_types = conf.get_recipe("inputs")

        self.con = rtde.RTDE(self.robot_ip, self.robot_port)
        self.con.connect()

        self.con.get_controller_version()

        if not self.con.send_output_setup(output_names, output_types):
            raise RuntimeError("Failed to set up outputs")
        
        self.inputs = self.con.send_input_setup(input_names, input_types)
        if self.inputs is None:
            raise RuntimeError("Failed to set up inputs")
        
        if not self.con.send_start():
            raise RuntimeError("Failed to start RTDE communication")
        
    def send_pose(self, x_mm, y_mm, angle_deg, valid):
        if self.con is None or self.inputs is None:
            raise RuntimeError("Not connected to robot")
        
        self.inputs.input_double_register_0 = round(x_mm/1000, 4)
        self.inputs.input_double_register_1 = round(y_mm/1000, 4)
        self.inputs.input_double_register_2 = math.radians(angle_deg)
        self.inputs.input_double_register_3 = float(valid)

        ok = self.con.send(self.inputs)
        if not ok:
            raise RuntimeError("Failed to send inputs to robot")
    
    def close(self):
        if self.con:
            try:
                self.con.send_pause()
            except Exception:
                pass
            try:
                self.con.disconnect()
            except Exception:
                pass