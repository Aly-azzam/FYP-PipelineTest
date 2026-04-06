import cv2


def extract_first_frame(video_path, output_image_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la vidéo.")
        return False

    success, frame = cap.read()

    if not success:
        print("Erreur : impossible de lire la première frame.")
        cap.release()
        return False

    cv2.imwrite(output_image_path, frame)
    cap.release()

    print(f"Première frame sauvegardée dans : {output_image_path}")
    return True