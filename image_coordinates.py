import cv2
import mediapipe as mp
import face_recognition

img_input = face_recognition.load_image_file("images/8.png")

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

while True:
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(img_input)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img_input, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
            imageLms = faceLms
            print(imageLms)
            # for id, lm in enumerate(faceLms.landmark):
            #     ih, iw, ic = img_input.shape
            #     p, s = int(lm.x * iw), int(lm.y * ih)
            #     print("picture in the database", id, p, s)
            #     img_coordinates = (id, p, s)

    cv2.imshow("image", img_input)
    cv2.waitKey(1)
