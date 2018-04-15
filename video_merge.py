import cv2
import numpy as np

kitti = "kitti.avi"
resnet ="Resetnet_coco.avi"
v1_coco ="v1_coco.avi"
v2_coco = "v2_coco.avi"

if __name__ == '__main__':
    cap_1= cv2.VideoCapture(kitti)
    cap_2 = cv2.VideoCapture(resnet)
    cap_3 = cv2.VideoCapture(v1_coco)
    cap_4 = cv2.VideoCapture(v2_coco)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter('result_final.avi', fourcc, cap_1.get(cv2.CAP_PROP_FPS),
                                   (int(cap_1.get(3)), int(cap_1.get(4))))

    width= int(cap_1.get(4))
    height = int(cap_1.get(3))
    result = np.zeros((2*width,2*height,3),np.uint8)
    cap_open = cap_1.isOpened() and cap_2.isOpened() and cap_3.isOpened() and cap_4.isOpened()
    if cap_open:
        while True:
            ret_1, frame_1 = cap_1.read()
            ret_2, frame_2 = cap_2.read()
            ret_3, frame_3 = cap_3.read()
            ret_4, frame_4 = cap_4.read()

            result[:width,:height,:] = frame_1;
            result[width:, :height, :] = frame_2;
            result[:width, height:, :] = frame_3;
            result[width:, height:, :] = frame_4;
            result_final =cv2.resize(result,(1280,720))
            video_writer.write(result_final)
            #cv2.imshow("frame1",frame_1)
            #cv2.imshow("frame2", frame_2)
            #cv2.imshow("frame3", frame_3)
            #cv2.imshow("frame4", frame_4)
            cv2.imshow("result",result.astype(np.uint8))


            if cv2.waitKey(1)& 0xFF == ord("q"):
                break
        cap_1.release()
        cap_2.release()
        cap_3.release()
        cap_4.release()
        video_writer.release()
        cv2.destroyAllWindows()