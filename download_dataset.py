import os
import requests
import zipfile
import hashlib
import logging
import tarfile
from typing import List, Optional, Dict
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "bold5000_download.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("bold5000_downloader")

# --- Configuration ---
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "BOLD5000_data")
FIGSHARE_BASE_URL = "https://figshare.com/ndownloader/files/"

# Alternative direct download URL for the complete dataset
DIRECT_DOWNLOAD_URL = "https://figshare.com/ndownloader/articles/14456124?private_link=bbaf45dca1b1b873ddfa"
DATASET_ARCHIVE = f"{OUTPUT_DIR}/BOLD5000_complete.tar.gz"

# Full file IDs from Figshare (BOLD5000 v2.0)
# These are the file IDs for all sessions of all subjects
FILE_IDS: Dict[str, str] = {
    # CSI1 (all 15 sessions)
    "CSI1_ses01": "16934002", "CSI1_ses02": "16934005", "CSI1_ses03": "16934008",
    "CSI1_ses04": "16934011", "CSI1_ses05": "16934017", "CSI1_ses06": "16934020",
    "CSI1_ses07": "16934023", "CSI1_ses08": "16934026", "CSI1_ses09": "16934029",
    "CSI1_ses10": "16934035", "CSI1_ses11": "16934038", "CSI1_ses12": "16934041",
    "CSI1_ses13": "16934044", "CSI1_ses14": "16934047", "CSI1_ses15": "16934050",
    
    # CSI2 (all 15 sessions)
    "CSI2_ses01": "16934014", "CSI2_ses02": "16934053", "CSI2_ses03": "16934056",
    "CSI2_ses04": "16934059", "CSI2_ses05": "16934062", "CSI2_ses06": "16934065",
    "CSI2_ses07": "16934068", "CSI2_ses08": "16934071", "CSI2_ses09": "16934074",
    "CSI2_ses10": "16934077", "CSI2_ses11": "16934080", "CSI2_ses12": "16934083",
    "CSI2_ses13": "16934086", "CSI2_ses14": "16934089", "CSI2_ses15": "16934092",
    
    # CSI3 (all 15 sessions)
    "CSI3_ses01": "16934032", "CSI3_ses02": "16934095", "CSI3_ses03": "16934098",
    "CSI3_ses04": "16934101", "CSI3_ses05": "16934104", "CSI3_ses06": "16934107",
    "CSI3_ses07": "16934110", "CSI3_ses08": "16934113", "CSI3_ses09": "16934116",
    "CSI3_ses10": "16934119", "CSI3_ses11": "16934122", "CSI3_ses12": "16934125",
    "CSI3_ses13": "16934128", "CSI3_ses14": "16934131", "CSI3_ses15": "16934134",
    
    # CSI4 (9 sessions)
    "CSI4_ses01": "16934137", "CSI4_ses02": "16934140", "CSI4_ses03": "16934143",
    "CSI4_ses04": "16934146", "CSI4_ses05": "16934149", "CSI4_ses06": "16934152",
    "CSI4_ses07": "16934155", "CSI4_ses08": "16934158", "CSI4_ses09": "16934161",
    
    # Metadata
    "metadata": "16934170"
}

# This is the URL for the stimulus images from BOLD5000.org
IMAGE_URL = "http://BOLD5000.org/downloads/Scene_Stimuli.zip"

# Expected file sizes and checksums for validation
FILE_CHECKSUMS = {
    "metadata": "5a3b22d0f9c6e230e58a3c9c0a18b54c"  # Example MD5 checksum
    # Add more checksums as needed for other files
}

def calculate_md5(file_path: str) -> str:
    """Calculate MD5 checksum for a file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def validate_file(file_path: str, expected_checksum: Optional[str] = None) -> bool:
    """Validate a file's integrity using MD5 checksum"""
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        return False
        
    if expected_checksum:
        actual_checksum = calculate_md5(file_path)
        if actual_checksum != expected_checksum:
            logger.error(f"Checksum mismatch for {file_path}. Expected: {expected_checksum}, Got: {actual_checksum}")
            return False
            
    return True

def download_file(url: str, destination: str, expected_checksum: Optional[str] = None) -> bool:
    """
    Generic downloader with progress bar and checksum validation
    
    Args:
        url: URL to download from
        destination: Local path to save the file
        expected_checksum: Optional MD5 checksum for validation
        
    Returns:
        bool: True if download was successful and passed validation
    """
    # Check if file already exists and is valid
    if os.path.exists(destination):
        logger.info(f"File already exists: {destination}")
        if expected_checksum and not validate_file(destination, expected_checksum):
            logger.warning(f"Existing file failed validation, re-downloading: {destination}")
        else:
            return True
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    try:
        # Stream download with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='iB',
            unit_scale=True
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
        
        # Validate downloaded file
        if expected_checksum and not validate_file(destination, expected_checksum):
            logger.error(f"Downloaded file failed validation: {destination}")
            return False
            
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed for {url}: {str(e)}")
        # Clean up partial download
        if os.path.exists(destination):
            os.remove(destination)
        return False

def setup_bold5000_direct_download(selected_subjects: Optional[List[str]] = None,
                                  selected_sessions: Optional[List[str]] = None,
                                  force_download: bool = False) -> bool:
    """
    Download and setup the BOLD5000 dataset using direct download link
    
    Args:
        selected_subjects: List of subjects to filter after extraction (default: all)
        selected_sessions: List of sessions to filter after extraction (default: all)
        force_download: If True, re-download files even if they exist
        
    Returns:
        bool: True if setup was successful
    """
    # Setup directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Download the complete dataset
    if force_download or not os.path.exists(DATASET_ARCHIVE):
        logger.info("Downloading complete BOLD5000 dataset (121 GB)...")
        logger.info("This will take a significant amount of time depending on your connection speed.")
        
        if not download_file(DIRECT_DOWNLOAD_URL, DATASET_ARCHIVE):
            logger.error("Failed to download dataset. Aborting setup.")
            return False
    
    # Extract the dataset
    if not os.path.exists(f"{OUTPUT_DIR}/GLMbetas"):
        logger.info("Extracting dataset files (this may take a while)...")
        try:
            # Check if it's a tar.gz file
            if tarfile.is_tarfile(DATASET_ARCHIVE):
                with tarfile.open(DATASET_ARCHIVE, 'r') as tar:
                    # Extract all or filtered members
                    members = tar.getmembers()
                    
                    # Filter members if subjects/sessions are specified
                    if selected_subjects or selected_sessions:
                        filtered_members = []
                        for member in members:
                            # Check if this member matches the requested subjects/sessions
                            should_extract = True
                            
                            if selected_subjects:
                                if not any(subj in member.name for subj in selected_subjects):
                                    should_extract = False
                            
                            if selected_sessions and should_extract:
                                if not any(sess in member.name for sess in selected_sessions):
                                    should_extract = False
                            
                            # Always extract metadata and stimuli
                            if "metadata" in member.name or "stimuli" in member.name:
                                should_extract = True
                                
                            if should_extract:
                                filtered_members.append(member)
                                
                        members = filtered_members
                    
                    # Extract the selected members with progress
                    for member in tqdm(members, desc="Extracting files"):
                        tar.extract(member, path=OUTPUT_DIR)
            # Try zip file if not tar
            elif zipfile.is_zipfile(DATASET_ARCHIVE):
                with zipfile.ZipFile(DATASET_ARCHIVE, 'r') as zip_ref:
                    # Similar filtering logic could be applied here
                    zip_ref.extractall(OUTPUT_DIR)
            else:
                logger.error(f"Unknown archive format for {DATASET_ARCHIVE}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to extract dataset: {str(e)}")
            return False
    
    return True

def setup_bold5000(selected_subjects: Optional[List[str]] = None, 
                  selected_sessions: Optional[List[str]] = None,
                  force_download: bool = False,
                  use_direct_download: bool = True) -> bool:
    """
    Download and setup the BOLD5000 dataset
    
    Args:
        selected_subjects: List of subjects to download (default: all)
        selected_sessions: List of sessions to download (default: all)
        force_download: If True, re-download files even if they exist
        use_direct_download: If True, use the direct download link instead of individual files
        
    Returns:
        bool: True if setup was successful
    """
    # Try direct download first if enabled
    if use_direct_download:
        logger.info("Using direct download method for complete dataset")
        if setup_bold5000_direct_download(selected_subjects, selected_sessions, force_download):
            # Download stimulus images if not included in the main archive
            stimuli_dir = f"{OUTPUT_DIR}/stimuli/Scene_Stimuli"
            if not os.path.exists(stimuli_dir):
                img_archive = f"{OUTPUT_DIR}/Scene_Stimuli.zip"
                if force_download or not os.path.exists(img_archive):
                    logger.info("Downloading stimulus images...")
                    if not download_file(IMAGE_URL, img_archive):
                        logger.error("Failed to download stimulus images")
                        return False
                    
                # Extract stimulus images
                logger.info("Extracting image files...")
                try:
                    with zipfile.ZipFile(img_archive, 'r') as zip_ref:
                        zip_ref.extractall(f"{OUTPUT_DIR}/stimuli")
                except zipfile.BadZipFile:
                    logger.error(f"Failed to extract {img_archive}. File may be corrupted.")
                    return False
            
            logger.info("Setup complete!")
            return True
        else:
            logger.warning("Direct download failed, falling back to individual file download method")
    
    # Fallback to original method
    # Setup directories
    os.makedirs(f"{OUTPUT_DIR}/GLMbetas", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/stimuli", exist_ok=True)
    
    if selected_subjects is None:
        selected_subjects = ["CSI1", "CSI2", "CSI3", "CSI4"]
    
    # Download metadata first
    metadata_dest = f"{OUTPUT_DIR}/trials_metadata.csv"
    metadata_checksum = FILE_CHECKSUMS.get("metadata")
    
    if force_download or not os.path.exists(metadata_dest):
        logger.info("Downloading metadata...")
        if not download_file(
            f"{FIGSHARE_BASE_URL}{FILE_IDS['metadata']}", 
            metadata_dest, 
            metadata_checksum
        ):
            logger.error("Failed to download metadata. Aborting setup.")
            return False
    
    # Download fMRI data
    logger.info(f"Downloading fMRI data for subjects: {selected_subjects}")
    download_success = True
    
    for name, file_id in FILE_IDS.items():
        if "metadata" in name:
            continue
            
        # Parse subject and session from the name
        parts = name.split('_')
        if len(parts) != 2:
            continue
            
        subject, session = parts
        
        # Check if this subject/session should be downloaded
        if subject not in selected_subjects:
            continue
            
        if selected_sessions is not None and session not in selected_sessions:
            continue
        
        dest = f"{OUTPUT_DIR}/GLMbetas/{name}_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR.nii.gz"
        checksum = FILE_CHECKSUMS.get(name)
        
        # Skip if already downloaded and not forced
        if not force_download and os.path.exists(dest) and (checksum is None or validate_file(dest, checksum)):
            logger.info(f"File already exists and is valid: {dest}")
            continue
            
        logger.info(f"Downloading {name}...")
        url = f"{FIGSHARE_BASE_URL}{file_id}"
        if not download_file(url, dest, checksum):
            logger.error(f"Failed to download {name}")
            download_success = False
            
    # Download stimulus images
    img_archive = f"{OUTPUT_DIR}/Scene_Stimuli.zip"
    if force_download or not os.path.exists(img_archive):
        logger.info("Downloading stimulus images...")
        if not download_file(IMAGE_URL, img_archive):
            logger.error("Failed to download stimulus images")
            download_success = False
        
    # Extract stimulus images
    stimuli_dir = f"{OUTPUT_DIR}/stimuli/Scene_Stimuli"
    if not os.path.exists(stimuli_dir):
        logger.info("Extracting image files...")
        try:
            with zipfile.ZipFile(img_archive, 'r') as zip_ref:
                zip_ref.extractall(f"{OUTPUT_DIR}/stimuli")
        except zipfile.BadZipFile:
            logger.error(f"Failed to extract {img_archive}. File may be corrupted.")
            download_success = False
    
    if download_success:
        logger.info("Setup complete!")
    else:
        logger.warning("Setup completed with some errors. Check the log for details.")
    
    return download_success

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download BOLD5000 dataset")
    parser.add_argument('--subjects', nargs='+', help='List of subjects to download')
    parser.add_argument('--sessions', nargs='+', help='List of sessions to download')
    parser.add_argument('--force', action='store_true', help='Force re-download of existing files')
    parser.add_argument('--direct', action='store_true', default=True, 
                       help='Use direct download method (default: True)')
    
    args = parser.parse_args()
    
    setup_bold5000(
        selected_subjects=args.subjects,
        selected_sessions=args.sessions,
        force_download=args.force,
        use_direct_download=args.direct
    )