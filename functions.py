from typing import List, Tuple, Optional
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from tqdm import tqdm
from scipy.ndimage import rotate
import random
from sys import maxsize as max_int
from sklearn.preprocessing import LabelEncoder


### Constants ###
image_size = 96
num_samples = 5
epochs = 50

label_encoder = LabelEncoder()
label_encoder.fit(["baseball", "basketball", "billiard_ball", "bowling_ball", "cricket_ball", "football", "golf_ball", "hockey_ball", "table_tennis_ball", "tennis_ball", "volleyball"])

label_map = {
    0: "baseball",
    1: "basketball",
    2: "billiard_ball",
    3: "bowling_ball",
    4: "cricket_ball",
    5: "football",
    6: "golf_ball",
    7: "hockey_ball",
    8: "table_tennis_ball",
    9: "tennis_ball",
    10: "volleyball"
}

### Image Functions ###

# Function to load images and labels from a directory
def load_images(folder: str, n: Optional[int] = None) -> Tuple[List[np.ndarray], List[str]]:
    images = []
    labels = []
    label_folders = os.listdir(folder)
    for label_folder in tqdm(label_folders):
        count = 0
        label_path = os.path.join(folder, label_folder)
        if os.path.isdir(label_path):
            for img_file in glob(os.path.join(label_path, "*.jpg")):
                if n is not None and count >= n:
                    continue
                img = cv2.imread(img_file, cv2.IMREAD_COLOR)
                images.append(img)
                labels.append(label_folder)
                count += 1

    assert len(images) == len(labels), "Mismatch in number of images and labels"
    return images, labels


def display_samples(images: List[np.ndarray], labels: List[str]) -> None:
    # display some sample images with their labels
    plt.figure(figsize=(10, 2))
    for i in range(num_samples):
        rand = np.random.randint(0, len(images))
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(cv2.cvtColor(images[rand], cv2.COLOR_BGR2RGB))
        plt.title(f"{labels[rand]}")
        plt.axis('off')
    plt.show()


### Image Processing Functions ###

def downsample_image(image: np.ndarray) -> np.ndarray:
    # Downsample the image if its dimensions are larger than 1000
    max_dim = 1000
    if max(image.shape) > max_dim:
        scale = max_dim / max(image.shape)
        new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image


# todo improve circle detection
def detect_circles(image: np.ndarray, max_circles = max_int, threshold = 100, show: bool = False) -> np.ndarray:
    circles: Optional[np.ndarray] = None
    threshold = 100
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    downsampled_image = downsample_image(image)
    blur = min(downsampled_image.shape) // 100
    blurred_image = cv2.blur(downsampled_image, (blur, blur))
    edge_image = cv2.Canny(blurred_image, threshold1=100, threshold2=150)
    while True:
        circles = cv2.HoughCircles(
            edge_image,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min(blurred_image.shape) // 5,
            param1=threshold,
            param2=threshold,
            minRadius=10,
            maxRadius=min(blurred_image.shape)
        )
        if circles is not None or threshold <= 5:
            break
        else:
            threshold -= 5
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
    else:
        circles = np.array([])

    if show:
        circle_image = image.copy()
        if circles is not None:
            for (x, y, r) in circles[:max_circles]:
                cv2.circle(circle_image, (x, y), r, (0, 255, 0), 4)

        image_list = {
            "Original": image,
            "Blurred": blurred_image,
            "Edges": edge_image,
            "Detected Circles": circle_image
        }

        plt.figure(figsize=(20, 4))
        for i, (name, img) in enumerate(image_list.items()):
            plt.subplot(1, len(image_list), i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(name)
            plt.axis('off')

        plt.show()

    return circles[:max_circles]


def crop_to_circle(img: np.ndarray, circles: np.ndarray) -> np.ndarray:
    cropped_imgs = []
    if circles is not None:
        if type(circles[0]) == int:
            x, y, r = circles
            cropped_img = img[y-r:y+r, x-r:x+r]
            cropped_imgs.append(cropped_img)
        else:
            for (x, y, r) in circles:
                cropped_img = img[y-r:y+r, x-r:x+r]
                cropped_imgs.append(cropped_img)
    return cropped_imgs


def remove_nones(images: List[np.ndarray], labels: List[str]) -> Tuple[np.ndarray, List[str]]:
    new_images = []
    new_labels = []
    if labels is None:
        labels = [None] * len(images)
    for img, label in zip(images, labels):
        if img is not None and img.shape[0] >= image_size and img.shape[1] >= image_size:
            new_images.append(img)
            new_labels.append(label)
    return np.array(new_images), new_labels


def preprocess_images(images: List[np.ndarray], labels: List[str], max: int = max_int) -> Tuple[np.ndarray, List[str]]:
    cropped_images = []
    cropped_labels = []
    images_with_labels = list(zip(images, labels))
    for img, label in tqdm(images_with_labels):
        circles = detect_circles(img, max_circles=max)
        cis = crop_to_circle(img, circles)
        cropped_images.extend(cis)
        cropped_labels.extend([label] * len(cis))

    processed_images, processed_labels = downsample_images(cropped_images, cropped_labels)
    return np.array(processed_images), processed_labels


def process_images(images: List[np.ndarray], labels: List[str], max: int = max_int) -> Tuple[List[np.ndarray], List[str]]:
    cropped_images = []
    for img in tqdm(images):
        circles = detect_circles(img, max_circles=max)
        circle_images = crop_to_circle(img, circles)
        cropped_images.append(circle_images)

    processed_images = []
    processed_labels = []
    for imgs, label in zip(cropped_images, labels):
        d_images, _ = downsample_images(imgs, [label * len(imgs)])
        if len(d_images) > 0:
            d_images = circle_mask_images(d_images)
            processed_images.append(d_images)
            processed_labels.append(label)

    return processed_images, processed_labels


def downsample_images(images: List[np.ndarray], labels: List[str] = None, size: int = image_size) -> Tuple[np.ndarray, List[str]]:
    downsampled_images = []
    for img in images:
        if img is None or img.shape[0] < size or img.shape[1] < size:
            downsampled_images.append(None)
        else:
            downsampled_images.append(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA))
    
    downsampled_images, labels = remove_nones(downsampled_images, labels)
    return np.array(downsampled_images), labels


def circle_mask_images(images: List[np.ndarray]) -> np.ndarray:
    masked_images = []
    for image in images:
        # Get the dimensions of the image
        height, width = image.shape[:2]

        # Create a mask with a filled circle in the center
        mask = np.zeros((height, width), dtype=np.uint8)
        center = (width // 2, height // 2)
        radius = min(center[0], center[1], width - center[0], height - center[1])
        cv2.circle(mask, center, radius, 255, -1)

        # Apply the mask to the image
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        masked_images.append(masked_image)

    return np.array(masked_images)


def augment_image(image):
    # randomly rotate the image
    angle = random.uniform(-45, 45)
    rotated_image = rotate(image, angle, reshape=False)

    # randomly flip the image
    if random.choice([True, False]):
        flipped_image = np.fliplr(rotated_image)
    else:
        flipped_image = np.flipud(rotated_image)

    # pad to original size and return
    return flipped_image


def diversify_images(images: np.ndarray, labels: List[str], times: int = 1) -> Tuple[np.ndarray, List[str]]:
    augmented_images = []
    augmented_labels = []
    images_with_labels = list(zip(images, labels))
    for img, label in tqdm(images_with_labels):
        # add the original image
        augmented_images.append(img)
        augmented_labels.append(label)
        # add the augmented images
        for _ in range(times):
            augmented_images.append(augment_image(img))
            augmented_labels.append(label)

    return np.array(augmented_images), augmented_labels