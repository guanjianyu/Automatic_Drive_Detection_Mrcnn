import numpy as np
import cv2
from lane_detection.calibration import calib, undistort
from lane_detection.threshold import gradient_combine, hls_combine, comb_result
from lane_detection.finding_lines import Line, warp_image, find_LR_lines, draw_lane, print_road_status, print_road_map


left_line = Line()
right_line = Line()

th_sobelx, th_sobely, th_mag, th_dir = (35, 100), (30, 255), (30, 255), (0.7, 1.3)
th_h, th_l, th_s = (10, 100), (0, 60), (85, 255)

# camera matrix & distortion coefficient
mtx, dist = calib()


def lane_detection(undist_img):
    undist_img = cv2.resize(undist_img, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA)
    rows, cols = undist_img.shape[:2]

    combined_gradient = gradient_combine(undist_img, th_sobelx, th_sobely, th_mag, th_dir)
    combined_hls = hls_combine(undist_img, th_h, th_l, th_s)
    combined_result = comb_result(combined_gradient, combined_hls)

    c_rows, c_cols = combined_result.shape[:2]
    s_LTop2, s_RTop2 = [c_cols / 2 - 24, 5], [c_cols / 2 + 24, 5]
    s_LBot2, s_RBot2 = [110, c_rows], [c_cols - 110, c_rows]

    src = np.float32([s_LBot2, s_LTop2, s_RTop2, s_RBot2])
    dst = np.float32([(170, 720), (170, 0), (550, 0), (550, 720)])

    warp_img, M, Minv = warp_image(combined_result, src, dst, (720, 720))
    searching_img = find_LR_lines(warp_img, left_line, right_line)
    w_comb_result, w_color_result = draw_lane(searching_img, left_line, right_line)

    # Drawing the lines back down onto the road
    color_result = cv2.warpPerspective(w_color_result, Minv, (c_cols, c_rows))
    lane_color = np.zeros_like(undist_img)
    lane_color[220:rows - 12, 0:cols] = color_result
    #result = cv2.addWeighted(undist_img, 1, lane_color, 0.3, 0)
    lane_color = cv2.resize(lane_color,None,fx=2,fy=2,interpolation=cv2.INTER_AREA)
    return lane_color

if __name__ == '__main__':
    pass