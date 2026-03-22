from pick_place_env import PickPlaceEnv
import cv2
import time
env = PickPlaceEnv(gui=True)

for _ in range(50000):
    env.move_ee(0.001, 0, 0)
    env.step_sim(10)
    img = env.get_camera_image()
    # cv2.imshow("camera", img)
    # cv2.waitKey(0)

# img = env.get_camera_image()
# cv2.imshow("camera", img)
# cv2.waitKey(0)

