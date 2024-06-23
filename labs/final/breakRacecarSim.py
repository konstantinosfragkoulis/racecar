import sys

sys.path.insert(0, "../../library")
import racecar_core
import subprocess
########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

def start():
    cmd_str = "brave https://piped.projectsegfau.lt/watch?v=dQw4w9WgXcQ?autoplay=1"
    subprocess.run(cmd_str, shell=True)

def update():
    
    for i in range(1, 500):
        if i == 499:
            print("i:", i)
        for j in range(1, 200):
            if j == 199 and i == 499:
                print("j:", j)
            for k in range(1, 200):
                if k == 199 and j == 99 and i == 499:
                    print("k:", k)
                var = (i/j/k)

    print(rc.get_delta_time())

    rc.drive.set_speed_angle(1, 0)

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()