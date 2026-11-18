import os
import cv2
import numpy as np
from mtcnn import MTCNN
from deepface.models.facial_recognition.Facenet import FaceNet128dClient

# ==============================
# PATHS
# ==============================
DATASET_PATH = r"E:\facieal_recoginition_projects\07_mtcnn\dataset\Khushboo"

# Duplicate copies excluded for fair comparison
EXCLUDED_IMAGES = {"18.jpg", "31.jpg", "45.jpg"}

# ==============================
# INITIALIZE MODELS
# ==============================
print("Loading MTCNN...")
detector = MTCNN()

print("Loading FaceNet...")
facenet = FaceNet128dClient()

print("Models loaded successfully!\n")


# ==============================
# FUNCTION: EXTRACT FACE
# ==============================
def extract_face(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return None

    # MTCNN expects RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detections = detector.detect_faces(rgb_image)

    if len(detections) == 0:
        return None

    # Select the largest detected face
    detection = max(
        detections,
        key=lambda d: d["box"][2] * d["box"][3]
    )

    x, y, w, h = detection["box"]

    # Prevent coordinates from going outside image
    x = max(0, x)
    y = max(0, y)

    x2 = min(rgb_image.shape[1], x + w)
    y2 = min(rgb_image.shape[0], y + h)

    face = rgb_image[y:y2, x:x2]

    if face.size == 0:
        return None

    # Resize for FaceNet
    face = cv2.resize(face, (160, 160))

    # Normalize
    face = face.astype("float32") / 255.0

    # Add batch dimension
    face = np.expand_dims(face, axis=0)

    return face


# ==============================
# FUNCTION: GET FACENET EMBEDDING
# ==============================
def get_embedding(image_path):

    face = extract_face(image_path)

    if face is None:
        return None

    embedding = facenet.forward(face)

    embedding = np.asarray(embedding).flatten()

    return embedding


# ==============================
# LOAD IMAGES
# ==============================
image_files = sorted(
    [
        f for f in os.listdir(DATASET_PATH)
        if f.lower().endswith(".jpg")
        and f not in EXCLUDED_IMAGES
    ],
    key=lambda x: int(os.path.splitext(x)[0])
)

print("Images used:", len(image_files))
print("Excluded:", sorted(EXCLUDED_IMAGES))
print()


# ==============================
# GENERATE EMBEDDINGS
# ==============================
embeddings = {}

for i, filename in enumerate(image_files, start=1):

    image_path = os.path.join(DATASET_PATH, filename)

    embedding = get_embedding(image_path)

    if embedding is not None:
        embeddings[filename] = embedding

    print(
        f"[{i}/{len(image_files)}] "
        f"{filename} -> "
        f"{'Embedding generated' if embedding is not None else 'FAILED'}"
    )


print("\n==============================")
print("EMBEDDING SUMMARY")
print("==============================")
print("Images processed:", len(image_files))
print("Successful embeddings:", len(embeddings))
print("Embedding size:", len(next(iter(embeddings.values()))))


# ==============================
# COMPARE ALL PAIRS
# ==============================
results = []

filenames = list(embeddings.keys())

for i in range(len(filenames)):

    for j in range(i + 1, len(filenames)):

        img1 = filenames[i]
        img2 = filenames[j]

        emb1 = embeddings[img1]
        emb2 = embeddings[img2]

        # Euclidean distance
        euclidean = np.linalg.norm(emb1 - emb2)

        # Cosine similarity
        denominator = (
            np.linalg.norm(emb1) *
            np.linalg.norm(emb2)
        )

        cosine = np.dot(emb1, emb2) / denominator

        results.append(
            (img1, img2, euclidean, cosine)
        )


# ==============================
# STATISTICS
# ==============================
euclidean_values = np.array(
    [r[2] for r in results]
)

cosine_values = np.array(
    [r[3] for r in results]
)

print("\n==============================")
print("MTCNN + FACENET RESULTS")
print("==============================")

print("Images used:", len(filenames))
print("Pairs compared:", len(results))

print("\nEuclidean Distance:")
print(f"Minimum : {euclidean_values.min():.6f}")
print(f"Maximum : {euclidean_values.max():.6f}")
print(f"Average : {euclidean_values.mean():.6f}")
print(f"Std Dev : {euclidean_values.std():.6f}")

print("\nCosine Similarity:")
print(f"Minimum : {cosine_values.min():.6f}")
print(f"Maximum : {cosine_values.max():.6f}")
print(f"Average : {cosine_values.mean():.6f}")
print(f"Std Dev : {cosine_values.std():.6f}")


# ==============================
# MOST SIMILAR PAIR
# ==============================
most_similar = max(
    results,
    key=lambda x: x[3]
)

least_similar = min(
    results,
    key=lambda x: x[3]
)

print("\nMost Similar Pair:")
print(
    f"{most_similar[0]}, "
    f"{most_similar[1]}, "
    f"cosine {most_similar[3]:.6f}"
)

print("\nLeast Similar Pair:")
print(
    f"{least_similar[0]}, "
    f"{least_similar[1]}, "
    f"cosine {least_similar[3]:.6f}"
)

print("\nMTCNN + FaceNet experiment completed successfully!")