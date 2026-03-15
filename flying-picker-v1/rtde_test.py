import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
import socket
import math
import time

PORT = 5005
ROBOT_IP = "169.254.30.160"
ROBOT_PORT = 30004
CONFIG_FILE = "rtde_config.xml"

# ---------------- TCP SERVER ----------------
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("localhost", PORT))
server.listen(1)

print(f"RTDE sender server listening on port {PORT}...")
conn, addr = server.accept()
print(f"Connected by {addr}")

# ---------------- RTDE SETUP ----------------
conf = rtde_config.ConfigFile(CONFIG_FILE)

output_names, output_types = conf.get_recipe("outputs")
input_names, input_types = conf.get_recipe("inputs")

max_attempts = 10
attempt = 0

while attempt < max_attempts:
    try:
        print(f"Connecting to robot (attempt {attempt+1})...")
        con = rtde.RTDE(ROBOT_IP, ROBOT_PORT)
        con.connect()
        print("Connected!")
        break
    except TimeoutError:
        print("Timeout. Retrying...")
        attempt += 1
        time.sleep(3)

if attempt == max_attempts:
    print("Failed to connect after multiple attempts.")

con.send_output_setup(output_names, output_types)
inputs = con.send_input_setup(input_names, input_types)

con.send_start()

# ---------------- RECEIVE + SEND LOOP ----------------
buf = ""

while True:
    data = conn.recv(4096)
    if not data:
        print("Receiver connection closed")
        break

    buf += data.decode("utf-8")

    while "\n" in buf:
        line, buf = buf.split("\n", 1)
        line = line.strip()

        if not line:
            continue

        try:
            x_mm, y_mm, angle_deg = map(float, line.split(","))

            print(f"Received pose: x={x_mm:.3f} mm, y={y_mm:.3f} mm, angle={angle_deg:.3f} deg")

            # Convert units
            x = round(x_mm, 1)
            y = round(y_mm, 1)
            angle = math.radians(angle_deg)

            # Send to RTDE registers
            inputs.input_double_register_24 = x
            inputs.input_double_register_25 = y
            inputs.input_double_register_26 = angle

            con.send(inputs)

            print("Pose sent to robot.")

        except Exception as e:
            print(f"Invalid data format: '{line}' - {e}")

# ---------------- CLEANUP ----------------
con.send_pause()
con.disconnect()
conn.close()
server.close()