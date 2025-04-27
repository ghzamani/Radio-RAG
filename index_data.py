import faiss
import skimage, torch, torchvision
import numpy as np
import torchxrayvision as xrv
import pickle

from utills import load_test_bench


def xray_transform(img, transform):
    # Prepare the image:
    img = np.array(img)
    img = xrv.datasets.normalize(img, 255)  # convert 8-bit image to [-1024, 1024] range
    # TODO ? handle RGB images (as it is in their repo)
    img = img[None, ...]

    img = transform(img)
    img = torch.from_numpy(img)
    return img


def save_to_database(vector, db_name):
    label_vectors = np.array(vector).astype('float32')  # FAISS needs float32
    # Normalize vectors if you want cosine similarity
    label_vectors = label_vectors / np.linalg.norm(label_vectors, axis=1, keepdims=True)
    # Build FAISS index
    d = label_vectors.shape[1]  # dimension
    index = faiss.IndexFlatIP(d)  # inner product = cosine similarity if normalized
    index.add(label_vectors)
    # Save FAISS index
    faiss.write_index(index, db_name)
    return index


def main():
    # load dataset
    dataset = load_test_bench()
    # specify transform
    transform = torchvision.transforms.Compose(
        [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
    # load model
    # weight_path = "/mnt/disk2/ghazal.zamaninezhad/models/densenet121-res224-mimic_ch.pt"
    model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch",
                                cache_dir="/mnt/disk2/ghazal.zamaninezhad/hf_cache")
    # put model on gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    processed_images = []
    reports = []
    for sample in dataset['validation'].select(range(5)):
        transformed = xray_transform(sample['image'], transform)
        processed_images.append(transformed)
        reports.append(sample['report'])

    print()
    # batched_images = np.stack(processed_images, axis=0)
    # # predict labels for dataset
    # outputs = model(batched_images)
    # print(outputs.shape)
    # labels_index = save_to_database(outputs, "label_vector.index")
    #
    # with open("index_to_report.pkl", "wb") as f:
    #     pickle.dump(reports, f)


if __name__ == '__main__':
    main()
