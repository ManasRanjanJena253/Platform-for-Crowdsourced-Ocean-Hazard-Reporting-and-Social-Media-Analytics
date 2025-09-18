import os
import shutil
import tempfile
import uuid
import warnings
import cloudinary
import cloudinary.uploader
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.hash import bcrypt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

load_dotenv()
app = FastAPI()

client = AsyncIOMotorClient(host = "localhost", port = 27017)
db = client["Crowd_Sourced_Ocean_Hazard_Reporting"]
user_collection = db["user"]
hotspot_collection = db["hotspot"]
report_collection = db["reports"]

# Configuring cloudinary storage for image and report storing
cloudinary.config(
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key = os.getenv("CLOUDINARY_API_KEY"),
    api_secret = os.getenv("CLOUDINARY_API_SECRET")
)

async def upload_file(file: UploadFile, folder: str = "hazard_reports_by_user"):
    """
    Uploads a file (from FastAPI UploadFile) to Cloudinary and returns its secure URL.
    """
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete = False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Upload to Cloudinary
        result = cloudinary.uploader.upload(
            tmp_path,
            folder = folder,
            resource_type = "auto"  # auto-detects image/video/pdf
        )
        return result["secure_url"]

    except Exception as e:
        print(f"Cloudinary upload failed: {e}")
        return None


async def authorize_user(user_name: str, password):
    """
    Function to authorize the user
    :param user_name: The name of the user
    :param password: The password of the user
    :return: Boolean, whether the credentials match or not.
    """
    user = await user_collection.find_one({"user_name": user_name}, {"_id": 0, "user_id": 1, "hashed_pwd": 1})
    if user and bcrypt.verify(password, user["hashed_pwd"]):
        return user["user_id"]
    return False

async def generate_uuid():

    user_id = uuid.uuid4()
    uuid_check = await user_collection.find_one({"user_id": str(user_id)})

    if uuid_check:
        while uuid_check:
            user_id = uuid.uuid4()
            uuid_check = await user_collection.find_one({"user_id": str(user_id)})

        return str(user_id)

    else:
        return str(user_id)

async def generate_report_id():

    report_id = uuid.uuid4()
    uuid_check = await report_collection.find_one({"report_id": str(report_id)})

    if uuid_check:
        while uuid_check:
            report_id = uuid.uuid4()
            uuid_check = await report_collection.find_one({"report_id": str(report_id)})

        return str(report_id)

    else:
        return str(report_id)

def sentence_embeddings(text, svd, vectorizer):
    """
    Used for preprocessing the text and converting them into vector embeddings using Glove-twitter model.
    """
    vectors = vectorizer.transform([text])
    vectorized_text = svd.transform(vectors)

    return vectorized_text

warnings.filterwarnings('ignore')

try:
    import hdbscan

    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("HDBSCAN not installed. Install with: pip install hdbscan")


def find_optimal_eps(data, min_samples=5, percentile=90):
    """
    Find optimal eps value for DBSCAN using k-distance graph method
    """
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(data)
    distances, indices = neighbors_fit.kneighbors(data)

    distances = np.sort(distances[:, -1], axis=0)
    eps = np.percentile(distances, percentile)

    return eps


def cluster_data(data, method='auto', scale=True):

    # Convert to numpy array if needed
    if isinstance(data, pd.DataFrame):
        data = data.values
    data = np.array(data)

    # Scale data if requested
    if scale:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = data.copy()

    # Determine method
    if method == 'auto':
        method = 'hdbscan' if HDBSCAN_AVAILABLE else 'dbscan'

    # Perform clustering
    if method == 'hdbscan':
        labels, model = perform_hdbscan(data_scaled)
    else:
        labels, model = perform_dbscan(data_scaled)

    # Calculate metrics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    metrics = {}
    if n_clusters > 1:
        mask = labels != -1
        if mask.sum() > 0:
            try:
                metrics['silhouette'] = silhouette_score(data_scaled[mask], labels[mask])
                metrics['davies_bouldin'] = davies_bouldin_score(data_scaled[mask], labels[mask])
            except:
                metrics['silhouette'] = None
                metrics['davies_bouldin'] = None

    # Prepare results
    results = {
        'labels': labels,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'noise_ratio': n_noise / len(labels),
        'metrics': metrics,
        'model': model,
        'data_scaled': data_scaled if scale else None,
        'method': method
    }

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"CLUSTERING RESULTS ({method.upper()})")
    print(f"{'=' * 50}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise} ({results['noise_ratio']:.1%})")
    if metrics:
        if metrics.get('silhouette'):
            print(f"Silhouette Score: {metrics['silhouette']:.3f}")
        if metrics.get('davies_bouldin'):
            print(f"Davies-Bouldin Score: {metrics['davies_bouldin']:.3f}")
    print(f"{'=' * 50}\n")

    return results


def perform_dbscan(data, eps=None, min_samples=None, **kwargs):
    """
    Perform DBSCAN clustering with automatic parameter tuning
    """
    # Auto-tune parameters if not provided
    if min_samples is None:
        min_samples = max(5, int(np.log(len(data))))

    if eps is None:
        eps = find_optimal_eps(data, min_samples)
        print(f"Auto-selected eps: {eps:.3f}")

    # Perform clustering
    model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
    labels = model.fit_predict(data)

    return labels, model


def perform_hdbscan(data, min_cluster_size=None, min_samples=None, **kwargs):
    """
    Perform HDBSCAN clustering with automatic parameter tuning
    """
    if not HDBSCAN_AVAILABLE:
        print("HDBSCAN not available, falling back to DBSCAN")
        return perform_dbscan(data)

    # Auto-tune parameters if not provided
    if min_cluster_size is None:
        min_cluster_size = max(5, int(0.01 * len(data)))

    if min_samples is None:
        min_samples = min_cluster_size

    # Perform clustering
    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        **kwargs
    )
    labels = model.fit_predict(data)

    return labels, model
