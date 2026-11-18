from mtcnn import MTCNN
import cv2
import os


# ==========================================
# 1. CREATE MTCNN DETECTOR
# ==========================================

print("Loading MTCNN detector...")

detector = MTCNN()

print("MTCNN loaded successfully!")


# ==========================================
# 2. DATASET
# ==========================================

dataset_path = r"dataset\Khushboo"


# ==========================================
# 3. PROCESS IMAGES
# ==========================================

total_images = 0
faces_detected = 0


for i in range(1, 51):

    image_path = os.path.join(
        dataset_path,
        f"{i}.jpg"
    )

    image = cv2.imread(image_path)

    if image is None:

        print(f"Could not load: {i}.jpg")
        continue

    total_images += 1

    # Convert BGR → RGB
    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Detect faces
    detections = detector.detect_faces(
        image_rgb
    )

    print("\n----------------------------------------")
    print(f"Image: {i}.jpg")
    print(f"Faces detected: {len(detections)}")

    if len(detections) > 0:

        # Select largest face
        detection = max(
            detections,
            key=lambda d:
                d["box"][2] * d["box"][3]
        )

        box = detection["box"]
        confidence = detection["confidence"]
        keypoints = detection["keypoints"]

        print("\nBounding Box:")
        print("x:", box[0])
        print("y:", box[1])
        print("width:", box[2])
        print("height:", box[3])

        print("\nConfidence:")
        print(f"{confidence:.6f}")

        print("\nFacial Landmarks:")

        print(
            "Left Eye:",
            keypoints["left_eye"]
        )

        print(
            "Right Eye:",
            keypoints["right_eye"]
        )

        print(
            "Nose:",
            keypoints["nose"]
        )

        print(
            "Mouth Left:",
            keypoints["mouth_left"]
        )

        print(
            "Mouth Right:",
            keypoints["mouth_right"]
        )

        faces_detected += 1


# ==========================================
# 4. SUMMARY
# ==========================================

print("\n========================================")
print("        MTCNN LANDMARK RESULTS")
print("========================================")

print("Total images:", total_images)

print("Images with faces:", faces_detected)

print(
    "Detection rate:",
    f"{(faces_detected / total_images) * 100:.2f}%"
)