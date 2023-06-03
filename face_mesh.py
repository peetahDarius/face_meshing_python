import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

while True:
    # when it finds the video, it convert the live video into RGB format
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results1 = faceMesh.process(imgRGB)

    if results1.multi_face_landmarks:
        for faceLms in results1.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
            videoLms = faceLms
            print(videoLms)
            # for id, lmk in enumerate(faceLms.landmark):
            #     ih, iw, ic = img.shape
            #     a, b = int(lmk.x * iw), int(lmk.y * ih)
            #     print("video feed", id, a, b)
            #     feed_coordinates = (id, a, b)

    c_time = time.time()
    fps = 1 / (c_time - pTime)
    p_time = c_time
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow("image", img)
    cv2.waitKey(1)
