import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image

def default_transform(x):
    """
    Default transform for numpy arrays.
    Handles different array shapes and converts them to a torch tensor
    in the shape (T, C, H, W).
    """
    # x: (T, H, W, C) or (H, W, C) or (C, H, W) or (T, C, H, W)
    if x.ndim == 4 and x.shape[-1] == 5:
        # (T, H, W, 5) -> (T, 5, H, W)
        x = np.ascontiguousarray(x.transpose(0, 3, 1, 2))
    elif x.ndim == 3 and x.shape[0] == 5:
        # (5, H, W) -> (1, 5, H, W)
        x = x[None, ...]
    elif x.ndim == 3 and x.shape[-1] == 5:
        # (H, W, 5) -> (1, 5, H, W)
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None, ...])
    # Ensure output is a tensor
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x).float()
    return x

class MultiLabelDataset(Dataset):
    """Multi-label dataset for smoke/fire classification
    Label encoding:
    - only class 0 (smoke) → [1,0,0]
    - only class 1 (fire) → [0,1,0]
    - both class 0 and 1 → [1,1,0]
    - no classes (empty) → [0,0,1]
    """
    def __init__(self, npy_dir: str, labels_dir: str, transform=None, target_size=(224, 224)):
        self.npy_dir = Path(npy_dir)
        self.labels_dir = Path(labels_dir)
        self.target_size = target_size
        self.samples = []
        
        # Default transforms for 5-channel input (RGB + MHI + Flow)
        if transform is None:
            self.transform = default_transform
        else:
            self.transform = transform
        
        # Find all .npy files and their corresponding .txt files
        for npy_file in self.npy_dir.rglob("*.npy"):
            # Try to find corresponding .txt file
            txt_file = self.labels_dir / (npy_file.stem + ".txt")
            if txt_file.exists():
                self.samples.append((npy_file, txt_file))
            
        print(f"Found {len(self.samples)} samples in {npy_dir}")
    
    def _find_label_file(self, npy_path):
        """Find corresponding label file for a given .npy file."""
        # This method is no longer used with the new __init__ logic
        # but kept for potential backward compatibility needs.
        label_filename = npy_path.stem + ".txt"
        
        # Check in the specified labels directory first if available
        if hasattr(self, 'labels_dir') and self.labels_dir:
             txt_file = self.labels_dir / label_filename
             if txt_file.exists():
                 return txt_file

        # Fallback to old logic if needed
        txt_file = npy_path.with_suffix('.txt')
        if txt_file.exists():
            return txt_file
        
        # Check for a corresponding labels directory (e.g., flow -> flow_labels)
        data_subdir_name = npy_path.parent.name
        parent_dir = npy_path.parent.parent
        
        potential_label_dir_name = data_subdir_name + '_labels'
        labels_dir = parent_dir / potential_label_dir_name
        txt_file = labels_dir / label_filename
        if labels_dir.exists() and txt_file.exists():
            return txt_file

        # General 'labels' directory
        labels_dir = parent_dir / 'labels'
        txt_file = labels_dir / label_filename
        if labels_dir.exists() and txt_file.exists():
            return txt_file

        return None
    
    def __len__(self):
        return len(self.samples)
    
    def _convert_to_multi_label(self, txt_path):
        """Convert .txt file content to multi-label format
        Returns a 3-element list: [smoke, fire, none]
        """
        # Initialize label vector [smoke, fire, none]
        label_vector = [0, 0, 0]
        
        # Read the .txt file if it exists
        if txt_path.exists():
            with open(txt_path, 'r') as f:
                # Get unique class indices from the file
                class_indices = set()
                for line in f:
                    try:
                        class_idx = int(line.strip().split()[0])
                        class_indices.add(class_idx)
                    except (ValueError, IndexError):
                        continue
                
                # Set labels based on presence of classes
                if 0 in class_indices and 1 in class_indices:
                    # Both smoke and fire
                    label_vector = [1, 1, 0]
                elif 0 in class_indices:
                    # Only smoke
                    label_vector = [1, 0, 0]
                elif 1 in class_indices:
                    # Only fire
                    label_vector = [0, 1, 0]
                else:
                    # No classes found in file
                    label_vector = [0, 0, 1]
        else:
            # If .txt file doesn't exist, treat as none
            label_vector = [0, 0, 1]
            
        return label_vector
    
    def _process_5channel_data(self, data):
        """Process 5-channel data (RGB + MHI + Flow) to ensure correct format"""
        # Ensure data is in the correct shape (5, H, W)
        if len(data.shape) == 3:
            if data.shape[0] == 5:
                # Already in correct format (5, H, W)
                pass
            elif data.shape[2] == 5:
                # Convert from (H, W, 5) to (5, H, W)
                data = np.transpose(data, (2, 0, 1))
            else:
                raise ValueError(f"Unexpected data shape: {data.shape}")
        else:
            raise ValueError(f"Expected 3D array, got shape: {data.shape}")
        
        # Ensure data is in uint8 format for PIL conversion
        if data.dtype != np.uint8:
            # Normalize to 0-255 range if needed
            if data.max() <= 1.0:
                data = (data * 255).astype(np.uint8)
            else:
                data = data.astype(np.uint8)
        
        return data
    
    def __getitem__(self, idx):
        npy_path, txt_path = self.samples[idx]
        
        # Load the feature data (RGB+MHI+Flow)
        try:
            data = np.load(str(npy_path))
            
            # Apply transforms
            if self.transform:
                data = self.transform(data)
            else:
                data = torch.from_numpy(data).float()

            # Resize spatial dimensions (H, W) to target_size
            data = F.interpolate(data, size=self.target_size, mode='bilinear', align_corners=False)
            
            # Get multi-label vector
            labels = self._convert_to_multi_label(txt_path)
            labels = torch.tensor(labels, dtype=torch.float32)
            
            return data, labels
            
        except Exception as e:
            print(f"Error loading sample {npy_path}: {str(e)}")
            # Return a zero tensor with correct shape in case of error
            data_shape = (1, 5, self.target_size[0], self.target_size[1])
            return torch.zeros(data_shape), torch.tensor([0, 0, 1], dtype=torch.float32)

class MultiLabelDatasetWithImages(MultiLabelDataset):
    """Extended dataset class that can handle separate image files"""
    def __init__(self, image_dir: str, mhi_dir: str, flow_dir: str, labels_dir: str,
                 transform=None, target_size=(224, 224)):
        # Note: We are not calling super().__init__() to avoid its npy file search.
        # This class manages its own sample list from separate image/mhi/flow files.
        self.image_dir = Path(image_dir) if image_dir else None
        self.mhi_dir = Path(mhi_dir) if mhi_dir else None
        self.flow_dir = Path(flow_dir) if flow_dir else None
        self.labels_dir = Path(labels_dir) if labels_dir else None
        self.transform = transform
        self.target_size = target_size
        self.samples = []
        
        # Find all .jpg files and their corresponding data and label files
        if self.image_dir and self.mhi_dir and self.flow_dir and self.labels_dir:
            for img_file in self.image_dir.rglob("*.jpg"):
                base_name = img_file.stem
                mhi_file = self.mhi_dir / f"{base_name}.npy"
                flow_file = self.flow_dir / f"{base_name}.npy"
                txt_file = self.labels_dir / f"{base_name}.txt"
                
                if mhi_file.exists() and flow_file.exists() and txt_file.exists():
                    self.samples.append((img_file, mhi_file, flow_file, txt_file))
            
            print(f"Found {len(self.samples)} samples with separate files")

    def _find_label_file_for_image(self, img_path):
        # This method is kept for reference but the logic is now in __init__
        if self.labels_dir:
            txt_file = self.labels_dir / (img_path.stem + '.txt')
            if txt_file.exists():
                return txt_file
        return None
    
    def __getitem__(self, idx):
        # This method now needs to handle the case for separate files explicitly
        # as it doesn't use the parent's __getitem__ logic directly
        try:
            img_path, mhi_path, flow_path, txt_path = self.samples[idx]
            
            # Load and combine data from separate files
            data = self._load_separate_files(img_path, mhi_path, flow_path)
            
            # Get multi-label vector
            labels = self._convert_to_multi_label(txt_path)
            labels = torch.tensor(labels, dtype=torch.float32)
            
            return data, labels

        except Exception as e:
            print(f"Error loading sample {img_path}: {str(e)}")
            data_shape = (5, self.target_size[0], self.target_size[1])
            return torch.zeros(data_shape), torch.tensor([0, 0, 1], dtype=torch.float32)
    
    def _load_separate_files(self, img_path, mhi_path, flow_path):
        """Load RGB, MHI, and Flow from separate files"""
        # Load RGB image
        rgb_img = Image.open(img_path).convert('RGB')
        rgb_tensor = transforms.ToTensor()(rgb_img)
        
        # Load MHI
        mhi_data = np.load(str(mhi_path))
        if len(mhi_data.shape) == 2:
            mhi_data = mhi_data[np.newaxis, :, :]  # Add channel dimension
        mhi_tensor = torch.from_numpy(mhi_data).float()
        
        # Load Flow
        flow_data = np.load(str(flow_path))
        if len(flow_data.shape) == 2:
            flow_data = flow_data[np.newaxis, :, :]  # Add channel dimension
        flow_tensor = torch.from_numpy(flow_data).float()
        
        # Concatenate channels: RGB(3) + MHI(1) + Flow(1) = 5 channels
        combined = torch.cat([rgb_tensor, mhi_tensor, flow_tensor], dim=0)
        
        # Resize to target size
        combined = torch.nn.functional.interpolate(
            combined.unsqueeze(0), 
            size=self.target_size, 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        # Normalize
        combined = transforms.Normalize(
            mean=[0.485, 0.456, 0.406, 0.5, 0.5],
            std=[0.229, 0.224, 0.225, 0.5, 0.5]
        )(combined)
        
        return combined