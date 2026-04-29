import os
import fnmatch
import json

import h5py
import yaml
import cv2
import numpy as np

from configs.state_vec import STATE_VEC_IDX_MAPPING

LEFT_ARM = [STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(7)]
RIGHT_ARM = [STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)]

LEFT_HAND_5 = [STATE_VEC_IDX_MAPPING[f"left_gripper_joint_{i}_pos"] for i in range(5)]
RIGHT_HAND_5 = [STATE_VEC_IDX_MAPPING[f"right_gripper_joint_{i}_pos"] for i in range(5)]

LEFT_HAND_AUX = STATE_VEC_IDX_MAPPING["left_arm_joint_7_pos"]
RIGHT_HAND_AUX = STATE_VEC_IDX_MAPPING["right_arm_joint_7_pos"]

USED_STATE_INDICES = (
    LEFT_ARM
    + RIGHT_ARM
    + LEFT_HAND_5
    + [LEFT_HAND_AUX]
    + RIGHT_HAND_5
    + [RIGHT_HAND_AUX]
)


class HDF5VLADataset:
    """
    This class is used to sample episodes from the embododiment dataset
    stored in HDF5.
    """
    def __init__(self) -> None:
        # The path to the HDF5 dataset directory.
        # Each HDF5 file contains one episode.
        #
        # Priority:
        # 1) env `RDT_HDF5_DIR`
        # 2) default to a sibling `../traindata` (works for this demo repo layout)
        # 3) fallback to the custom finetune dataset path
        default_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "traindata"))
        HDF5_DIR = os.environ.get("RDT_HDF5_DIR", default_dir)
        if not os.path.isdir(HDF5_DIR):
            HDF5_DIR = "data/datasets/my_cool_dataset/rdt_data/"
        self.HDF5_DIR = HDF5_DIR
        self.DATASET_NAME = os.environ.get("RDT_DATASET_NAME", "my_cool_dataset")
        
        self.file_paths = []
        for root, _, files in os.walk(self.HDF5_DIR):
            for filename in fnmatch.filter(files, '*.hdf5'):
                file_path = os.path.join(root, filename)
                self.file_paths.append(file_path)
        if len(self.file_paths) == 0:
            raise FileNotFoundError(
                f"No .hdf5 episodes found under HDF5_DIR={self.HDF5_DIR}. "
                f"Set env RDT_HDF5_DIR or generate data into ../traindata."
            )
                
        # Load the config
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']
    
        # Get each episode's len
        episode_lens = []
        for file_path in self.file_paths:
            valid, res = self.parse_hdf5_file_state_only(file_path)
            _len = res['state'].shape[0] if valid else 0
            episode_lens.append(_len)
        total = float(np.sum(episode_lens))
        if total <= 0:
            raise RuntimeError("All episodes are invalid/empty after filtering.")
        self.episode_sample_weights = np.array(episode_lens, dtype=np.float64) / total
    
    def __len__(self):
        return len(self.file_paths)
    
    def get_dataset_name(self):
        return self.DATASET_NAME
    
    def get_item(self, index: int=None, state_only=False):
        """Get a training sample at a random timestep.

        Args:
            index (int, optional): the index of the episode.
                If not provided, a random episode will be selected.
            state_only (bool, optional): Whether to return only the state.
                In this way, the sample will contain a complete trajectory rather
                than a single timestep. Defaults to False.

        Returns:
           sample (dict): a dictionary containing the training sample.
        """
        while True:
            if index is None:
                file_path = np.random.choice(self.file_paths, p=self.episode_sample_weights)
            else:
                file_path = self.file_paths[index]
            valid, sample = self.parse_hdf5_file(file_path) \
                if not state_only else self.parse_hdf5_file_state_only(file_path)
            if valid:
                return sample
            else:
                index = np.random.randint(0, len(self.file_paths))

    def _fill_state_26(self, values: np.ndarray) -> np.ndarray:
        """
        Map raw 26D vectors into unified state vectors.

        Layout:
          0:7   -> left arm joints (7)
          7:14  -> right arm joints (7)
          14:20 -> left hand/gripper (6), where the 6th dim goes to left arm slot 7
          20:26 -> right hand/gripper (6), where the 6th dim goes to right arm slot 7
        """
        values = np.asarray(values, dtype=np.float32)
        if values.shape[-1] != 26:
            raise ValueError(f"Expected last dim=26, got {values.shape}")

        out = np.zeros(values.shape[:-1] + (self.STATE_DIM,), dtype=np.float32)

        out[..., LEFT_ARM] = values[..., 0:7]
        out[..., RIGHT_ARM] = values[..., 7:14]
        out[..., LEFT_HAND_5] = values[..., 14:19]
        out[..., LEFT_HAND_AUX] = values[..., 19]
        out[..., RIGHT_HAND_5] = values[..., 20:25]
        out[..., RIGHT_HAND_AUX] = values[..., 25]
        return out

    def _rdt128_to_raw26(self, values: np.ndarray) -> np.ndarray:
        """
        Inverse mapping of _fill_state_26.
        """
        values = np.asarray(values, dtype=np.float32)
        if values.shape[-1] != self.STATE_DIM:
            raise ValueError(f"Expected last dim={self.STATE_DIM}, got {values.shape}")

        out = np.zeros(values.shape[:-1] + (26,), dtype=np.float32)
        out[..., 0:7] = values[..., LEFT_ARM]
        out[..., 7:14] = values[..., RIGHT_ARM]
        out[..., 14:19] = values[..., LEFT_HAND_5]
        out[..., 19] = values[..., LEFT_HAND_AUX]
        out[..., 20:25] = values[..., RIGHT_HAND_5]
        out[..., 25] = values[..., RIGHT_HAND_AUX]
        return out
    
    def parse_hdf5_file(self, file_path):
        """[Modify] Parse a hdf5 file to generate a training sample at
            a random timestep.

        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "meta": {
                        "dataset_name": str,    # the name of your dataset.
                        "#steps": int,          # the number of steps in the episode,
                                                # also the total timesteps.
                        "instruction": str      # the language instruction for this episode.
                    },                           
                    "step_id": int,             # the index of the sampled step,
                                                # also the timestep t.
                    "state": ndarray,           # state[t], (1, STATE_DIM).
                    "state_std": ndarray,       # std(state[:]), (STATE_DIM,).
                    "state_mean": ndarray,      # mean(state[:]), (STATE_DIM,).
                    "state_norm": ndarray,      # norm(state[:]), (STATE_DIM,).
                    "actions": ndarray,         # action[t:t+CHUNK_SIZE], (CHUNK_SIZE, STATE_DIM).
                    "state_indicator", ndarray, # indicates the validness of each dim, (STATE_DIM,).
                    "cam_high": ndarray,        # external camera image, (IMG_HISORY_SIZE, H, W, 3)
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_high_mask": ndarray,   # indicates the validness of each timestep, (IMG_HISORY_SIZE,) boolean array.
                                                # For the first IMAGE_HISTORY_SIZE-1 timesteps, the mask should be False.
                    "cam_left_wrist": ndarray,  # left wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_left_wrist_mask": ndarray,
                    "cam_right_wrist": ndarray, # right wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                                                # If only one wrist, make it right wrist, plz.
                    "cam_right_wrist_mask": ndarray
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, 'r') as f:
            # Support both the original example layout and this demo layout.
            if 'observations' in f and 'qpos' in f['observations']:
                qpos = f['observations']['qpos'][:]
            else:
                qpos = f['observations/qpos'][:]
            num_steps = qpos.shape[0]
            if num_steps < 16:
                return False, None
            
            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            first_idx = int(indices[0]) if len(indices) > 0 else 1
            
            # We randomly sample a timestep
            step_id = np.random.randint(max(first_idx - 1, 0), num_steps)
            
            # Load instruction: HDF5 > official expanded instruction json > nearby text.
            instruction = self._load_instruction(f, file_path)
            
            # Assemble the meta
            meta = {
                "dataset_name": self.DATASET_NAME,
                "#steps": num_steps,
                "step_id": step_id,
                "instruction": instruction
            }
            
            qpos_raw = qpos.astype(np.float32, copy=False)

            # Read action.
            if 'action' in f:
                raw_action = f['action'][:]
            else:
                raw_action = f['actions'][:]
            action_raw = raw_action.astype(np.float32, copy=False)
            action_chunk_raw = action_raw[step_id:step_id + self.CHUNK_SIZE]
            
            # Parse the state and action
            state = self._fill_state_26(qpos_raw[step_id:step_id + 1])
            state_std = self._fill_state_26(np.std(qpos_raw, axis=0))
            state_mean = self._fill_state_26(np.mean(qpos_raw, axis=0))
            state_norm = self._fill_state_26(np.sqrt(np.mean(qpos_raw ** 2, axis=0)))
            actions = self._fill_state_26(action_chunk_raw)
            action_len = actions.shape[0]
            if action_len < self.CHUNK_SIZE:
                if action_len == 0:
                    return False, None
                pad = np.repeat(actions[-1:], self.CHUNK_SIZE - action_len, axis=0)
                actions = np.concatenate([actions, pad], axis=0)

            state_indicator = np.zeros((self.STATE_DIM,), dtype=np.float32)
            state_indicator[USED_STATE_INDICES] = 1.0
            
            # Parse the images
            # This demo dataset may not contain any images; return empty arrays so
            # the training pipeline will replace them with background images.
            has_images = ('observations' in f) and ('images' in f['observations'])
            if has_images and ('cam_high' in f['observations']['images']):
                def parse_img(key):
                    imgs = []
                    for i in range(max(step_id-self.IMG_HISORY_SIZE+1, 0), step_id+1):
                        img = f['observations']['images'][key][i]
                        decoded = self._decode_image(img, file_path=file_path, key=key, step=i)
                        imgs.append(decoded)
                    imgs = np.stack(imgs)
                    if imgs.shape[0] < self.IMG_HISORY_SIZE:
                        imgs = np.concatenate([
                            np.tile(imgs[:1], (self.IMG_HISORY_SIZE-imgs.shape[0], 1, 1, 1)),
                            imgs
                        ], axis=0)
                    return imgs
                cam_high = parse_img('cam_high')
                start_idx = max(first_idx - 1, 0)
                valid_len = min(step_id - start_idx + 1, self.IMG_HISORY_SIZE)
                cam_high_mask = np.array(
                    [False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len
                )
                cam_left_wrist = np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0), dtype=np.uint8)
                cam_left_wrist_mask = np.zeros((self.IMG_HISORY_SIZE,), dtype=bool)
                cam_right_wrist = np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0), dtype=np.uint8)
                cam_right_wrist_mask = np.zeros((self.IMG_HISORY_SIZE,), dtype=bool)
            else:
                cam_high = np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0), dtype=np.uint8)
                cam_left_wrist = np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0), dtype=np.uint8)
                cam_right_wrist = np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0), dtype=np.uint8)
                cam_high_mask = np.zeros((self.IMG_HISORY_SIZE,), dtype=bool)
                cam_left_wrist_mask = np.zeros((self.IMG_HISORY_SIZE,), dtype=bool)
                cam_right_wrist_mask = np.zeros((self.IMG_HISORY_SIZE,), dtype=bool)
            
            # Return the resulting sample
            # For unavailable images, return zero-shape arrays, i.e., (IMG_HISORY_SIZE, 0, 0, 0)
            # E.g., return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0)) for the key "cam_left_wrist",
            # if the left-wrist camera is unavailable on your robot
            return True, {
                "meta": meta,
                "state": state,
                "state_std": state_std,
                "state_mean": state_mean,
                "state_norm": state_norm,
                "actions": actions,
                "state_indicator": state_indicator,
                "cam_high": cam_high,
                "cam_high_mask": cam_high_mask,
                "cam_left_wrist": cam_left_wrist,
                "cam_left_wrist_mask": cam_left_wrist_mask,
                "cam_right_wrist": cam_right_wrist,
                "cam_right_wrist_mask": cam_right_wrist_mask
            }

    @staticmethod
    def _decode_hdf5_string(value):
        if isinstance(value, (bytes, np.bytes_)):
            return value.decode('utf-8')
        if isinstance(value, np.ndarray) and value.shape == ():
            scalar = value.item()
            if isinstance(scalar, (bytes, np.bytes_)):
                return scalar.decode('utf-8')
            return str(scalar)
        return str(value)

    def _load_instruction(self, hdf5_file, file_path: str) -> str:
        if 'instruction' in hdf5_file:
            return self._decode_hdf5_string(hdf5_file['instruction'][()]).strip()

        dir_path = os.path.dirname(file_path)
        json_path = os.path.join(dir_path, "expanded_instruction_gpt-4-turbo.json")
        if os.path.isfile(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                if isinstance(data, dict):
                    ins = data.get("instruction", "")
                    if ins:
                        return str(ins).strip()
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    ins = data[0].get("instruction", "")
                    if ins:
                        return str(ins).strip()
            except Exception:
                pass

        txt_path = os.path.join(dir_path, "instruction.txt")
        if os.path.isfile(txt_path):
            with open(txt_path, "r", encoding="utf-8") as fp:
                return fp.read().strip()
        return ""

    @staticmethod
    def _decode_image(img, file_path: str, key: str, step: int):
        # Support raw RGB uint8 arrays in HDF5 directly.
        if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[-1] == 3:
            return img.astype(np.uint8, copy=False)

        # Support encoded bytes/object datasets (e.g., JPEG/PNG).
        if isinstance(img, np.ndarray):
            buffer = img.tobytes()
        elif isinstance(img, (bytes, np.bytes_, np.void)):
            buffer = bytes(img)
        else:
            buffer = np.asarray(img).tobytes()

        decoded = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(f"Failed to decode image in {file_path}, key={key}, step={step}")
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    def parse_hdf5_file_state_only(self, file_path):
        """[Modify] Parse a hdf5 file to generate a state trajectory.

        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "state": ndarray,           # state[:], (T, STATE_DIM).
                    "action": ndarray,          # action[:], (T, STATE_DIM).
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, 'r') as f:
            if 'observations' in f and 'qpos' in f['observations']:
                qpos = f['observations']['qpos'][:]
            else:
                qpos = f['observations/qpos'][:]
            num_steps = qpos.shape[0]
            if num_steps < 16:
                return False, None
            
            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            first_idx = int(indices[0]) if len(indices) > 0 else 1
            
            if 'action' in f:
                target_qpos = f['action'][:]
            else:
                target_qpos = f['actions'][:]

            qpos_raw = qpos.astype(np.float32, copy=False)
            target_qpos = target_qpos.astype(np.float32, copy=False)
            
            # Parse the state and action
            start_idx = max(first_idx - 1, 0)
            state = self._fill_state_26(qpos_raw[start_idx:])
            action = self._fill_state_26(target_qpos[start_idx:])
            
            # Return the resulting sample
            return True, {
                "state": state,
                "action": action
            }

if __name__ == "__main__":
    ds = HDF5VLADataset()
    for i in range(len(ds)):
        print(f"Processing episode {i}/{len(ds)}...")
        ds.get_item(i)
