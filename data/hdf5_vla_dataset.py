import os
import fnmatch

import h5py
import yaml
import cv2
import numpy as np

from configs.state_vec import STATE_VEC_IDX_MAPPING


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
        # 3) fallback to original example path
        default_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "traindata"))
        HDF5_DIR = os.environ.get("RDT_HDF5_DIR", default_dir)
        if not os.path.isdir(HDF5_DIR):
            HDF5_DIR = "data/datasets/agilex/rdt_data/"
        self.HDF5_DIR = HDF5_DIR
        # Keep the default dataset name as `agilex` to reuse existing finetune configs.
        self.DATASET_NAME = os.environ.get("RDT_DATASET_NAME", "agilex")
        
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
            # Drop too-short episode (align with config defaults; avoid over-filtering small demos)
            if num_steps < 32:
                return False, None
            
            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no qpos that exceeds the threshold.")
            
            # We randomly sample a timestep
            step_id = np.random.randint(first_idx-1, num_steps)
            
            # Load instruction (prefer embedded in hdf5; fallback to nearby files)
            instruction = ""
            if 'instruction' in f:
                val = f['instruction'][()]
                instruction = val.decode('utf-8') if isinstance(val, (bytes, np.bytes_)) else str(val)
            else:
                dir_path = os.path.dirname(file_path)
                txt_path = os.path.join(dir_path, "instruction.txt")
                if os.path.isfile(txt_path):
                    with open(txt_path, "r", encoding="utf-8") as fp:
                        instruction = fp.read().strip()
                else:
                    # Last resort: keep empty instruction for robustness.
                    instruction = ""
            
            # Assemble the meta
            meta = {
                "dataset_name": self.DATASET_NAME,
                "#steps": num_steps,
                "step_id": step_id,
                "instruction": instruction
            }
            
            # Read action.
            if 'action' in f:
                raw_action = f['action'][:]
            else:
                raw_action = f['actions'][:]

            # Our demo raw format is 26D:
            #   left arm 7 | right arm 7 | left hand 6 | right hand 6
            # For stability, we lightly rescale hand values to ~[0,1] (they can be large in raw logs).
            def _rescale_hand(arr):
                arr = arr.astype(np.float32, copy=False)
                left_hand = arr[..., 14:20]
                right_hand = arr[..., 20:26]
                # If already small, keep; otherwise squash by a constant scale.
                hand_scale = float(os.environ.get("RDT_HAND_SCALE", "2000.0"))
                if np.nanmax(np.abs(left_hand)) > 10.0 or np.nanmax(np.abs(right_hand)) > 10.0:
                    left_hand = np.clip(left_hand / hand_scale, 0.0, 1.0)
                    right_hand = np.clip(right_hand / hand_scale, 0.0, 1.0)
                    arr = arr.copy()
                    arr[..., 14:20] = left_hand
                    arr[..., 20:26] = right_hand
                return arr

            qpos = _rescale_hand(qpos)
            raw_action = _rescale_hand(raw_action)
            target_qpos = raw_action[step_id:step_id + self.CHUNK_SIZE]
            
            # Parse the state and action
            state = qpos[step_id:step_id+1]
            state_std = np.std(qpos, axis=0)
            state_mean = np.mean(qpos, axis=0)
            state_norm = np.sqrt(np.mean(qpos**2, axis=0))
            actions = target_qpos
            if actions.shape[0] < self.CHUNK_SIZE:
                # Pad the actions using the last action
                actions = np.concatenate([
                    actions,
                    np.tile(actions[-1:], (self.CHUNK_SIZE-actions.shape[0], 1))
                ], axis=0)
            
            # Fill the state/action into the unified vector
            def fill_in_demo26(values: np.ndarray) -> np.ndarray:
                """
                Map demo 26D vector to RDT unified 128D vector.
                demo26 layout:
                  [0:7]   left arm joints (7)
                  [7:14]  right arm joints (7)
                  [14:20] left hand (6)  -> left gripper joints (5) with a simple reduction
                  [20:26] right hand (6) -> right gripper joints (5)
                """
                values = np.asarray(values, dtype=np.float32)
                uni = np.zeros(values.shape[:-1] + (self.STATE_DIM,), dtype=np.float32)

                left_arm = values[..., 0:7]
                right_arm = values[..., 7:14]
                left_hand = values[..., 14:20]
                right_hand = values[..., 20:26]

                # Arms: fill first 7 joint slots (out of 10 reserved)
                for i in range(7):
                    uni[..., STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"]] = left_arm[..., i]
                    uni[..., STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"]] = right_arm[..., i]

                # Hands: we have 6 dims but only 5 gripper_joint slots reserved.
                # We merge the last two dims by average to fit into 5.
                def hand6_to_5(hand6):
                    hand5 = np.zeros(hand6.shape[:-1] + (5,), dtype=np.float32)
                    hand5[..., 0:4] = hand6[..., 0:4]
                    hand5[..., 4] = 0.5 * (hand6[..., 4] + hand6[..., 5])
                    return hand5

                left_g = hand6_to_5(left_hand)
                right_g = hand6_to_5(right_hand)
                for i in range(5):
                    uni[..., STATE_VEC_IDX_MAPPING[f"left_gripper_joint_{i}_pos"]] = left_g[..., i]
                    uni[..., STATE_VEC_IDX_MAPPING[f"right_gripper_joint_{i}_pos"]] = right_g[..., i]

                return uni

            state = fill_in_demo26(state)
            state_indicator = fill_in_demo26(np.ones_like(state_std, dtype=np.float32))
            state_std = fill_in_demo26(state_std)
            state_mean = fill_in_demo26(state_mean)
            state_norm = fill_in_demo26(state_norm)
            actions = fill_in_demo26(actions)
            
            # Parse the images
            # This demo dataset may not contain any images; return empty arrays so
            # the training pipeline will replace them with background images.
            has_images = ('observations' in f) and ('images' in f['observations'])
            if has_images:
                def parse_img(key):
                    imgs = []
                    for i in range(max(step_id-self.IMG_HISORY_SIZE+1, 0), step_id+1):
                        img = f['observations']['images'][key][i]
                        imgs.append(cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR))
                    imgs = np.stack(imgs)
                    if imgs.shape[0] < self.IMG_HISORY_SIZE:
                        imgs = np.concatenate([
                            np.tile(imgs[:1], (self.IMG_HISORY_SIZE-imgs.shape[0], 1, 1, 1)),
                            imgs
                        ], axis=0)
                    return imgs
                cam_high = parse_img('cam_high')
                valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
                cam_high_mask = np.array(
                    [False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len
                )
                cam_left_wrist = parse_img('cam_left_wrist')
                cam_left_wrist_mask = cam_high_mask.copy()
                cam_right_wrist = parse_img('cam_right_wrist')
                cam_right_wrist_mask = cam_high_mask.copy()
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
            if num_steps < 32:
                return False, None
            
            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no qpos that exceeds the threshold.")
            
            if 'action' in f:
                target_qpos = f['action'][:]
            else:
                target_qpos = f['actions'][:]

            def _rescale_hand(arr):
                arr = arr.astype(np.float32, copy=False)
                left_hand = arr[..., 14:20]
                right_hand = arr[..., 20:26]
                hand_scale = float(os.environ.get("RDT_HAND_SCALE", "2000.0"))
                if np.nanmax(np.abs(left_hand)) > 10.0 or np.nanmax(np.abs(right_hand)) > 10.0:
                    left_hand = np.clip(left_hand / hand_scale, 0.0, 1.0)
                    right_hand = np.clip(right_hand / hand_scale, 0.0, 1.0)
                    arr = arr.copy()
                    arr[..., 14:20] = left_hand
                    arr[..., 20:26] = right_hand
                return arr

            qpos = _rescale_hand(qpos)
            target_qpos = _rescale_hand(target_qpos)
            
            # Parse the state and action
            state = qpos[first_idx-1:]
            action = target_qpos[first_idx-1:]
            
            # Fill the state/action into the unified vector
            def fill_in_demo26(values: np.ndarray) -> np.ndarray:
                values = np.asarray(values, dtype=np.float32)
                uni = np.zeros(values.shape[:-1] + (self.STATE_DIM,), dtype=np.float32)
                left_arm = values[..., 0:7]
                right_arm = values[..., 7:14]
                left_hand = values[..., 14:20]
                right_hand = values[..., 20:26]
                for i in range(7):
                    uni[..., STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"]] = left_arm[..., i]
                    uni[..., STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"]] = right_arm[..., i]
                def hand6_to_5(hand6):
                    hand5 = np.zeros(hand6.shape[:-1] + (5,), dtype=np.float32)
                    hand5[..., 0:4] = hand6[..., 0:4]
                    hand5[..., 4] = 0.5 * (hand6[..., 4] + hand6[..., 5])
                    return hand5
                left_g = hand6_to_5(left_hand)
                right_g = hand6_to_5(right_hand)
                for i in range(5):
                    uni[..., STATE_VEC_IDX_MAPPING[f"left_gripper_joint_{i}_pos"]] = left_g[..., i]
                    uni[..., STATE_VEC_IDX_MAPPING[f"right_gripper_joint_{i}_pos"]] = right_g[..., i]
                return uni
            state = fill_in_demo26(state)
            action = fill_in_demo26(action)
            
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
