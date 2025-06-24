import argparse
from collections import Counter, defaultdict
import logging
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_default_model_and_env,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    print_and_save,
)
from calvin_agent.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint
import clip
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
import wandb
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from torchvision.io import write_video

from calvin_env.envs.play_table_env import get_env

logger = logging.getLogger(__name__)

EP_LEN = 360
NUM_SEQUENCES = 1000


def get_epoch(checkpoint: Path) -> str:
    """
    Extract the epoch number from a checkpoint filename.

    Args:
        checkpoint: Path to the checkpoint file

    Returns:
        The epoch number as a string, or "0" if no epoch is found in the filename
    """
    if "=" not in checkpoint.stem:
        return "0"
    return checkpoint.stem.split("=")[1]


def make_env(dataset_path: str) -> Any:
    """
    Create a CALVIN environment for evaluation.

    Args:
        dataset_path: Path to the dataset root directory

    Returns:
        A configured CALVIN environment instance
    """
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env


class CustomModel(CalvinBaseModel):
    def __init__(self) -> None:
        """
        Initialize a custom model interface.

        This is a base class that should be implemented by custom model architectures.
        """
        logger.warning("Please implement these methods as an interface to your custom model architecture.")
        raise NotImplementedError

    def reset(self) -> None:
        """
        Reset the model's internal state.

        This method is called at the beginning of each episode or sequence.
        """
        raise NotImplementedError

    def step(self, obs: Dict[str, Any], goal: str) -> Dict[str, Any]:
        """
        Take a step in the environment based on current observations and goal.

        Args:
            obs: Environment observations
            goal: Embedded language goal

        Returns:
            Dictionary containing the predicted action
        """
        raise NotImplementedError


def evaluate_policy(
    model: CustomModel,
    env: Any,
    epoch: str,
    eval_log_dir: Optional[str] = None,
    debug: bool = False,
    create_plan_tsne: bool = False,
) -> List[int]:
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch: Current epoch number for logging
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        List of success counts for each evaluated sequence
    """
    conf_dir = Path(__file__).absolute().parents[2] / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)

    eval_sequences = get_sequences(NUM_SEQUENCES)

    results = []
    plans = defaultdict(list)

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, plans, debug)
        results.append(result)
        if not debug:
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )

    if create_plan_tsne:
        create_tsne(plans, eval_log_dir, epoch)
    print_and_save(results, eval_sequences, eval_log_dir, epoch)

    return results


def evaluate_sequence(
    env: Any,
    model: CustomModel,
    task_checker: Any,
    initial_state: Any,
    eval_sequence: List[str],
    val_annotations: Dict[str, Any],
    plans: defaultdict,
    debug: bool,
) -> int:
    """
    Evaluates a sequence of language instructions.

    Args:
        env: The CALVIN environment
        model: The model to evaluate
        task_checker: Task oracle for checking task completion
        initial_state: Initial state of the environment
        eval_sequence: List of language instructions to evaluate
        val_annotations: Validation annotations for language goals
        plans: Dictionary to collect plan data for TSNE visualization
        debug: Whether to print debug information

    Returns:
        Number of successfully completed subtasks in the sequence
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask in eval_sequence:
        success = rollout(env, model, task_checker, subtask, val_annotations, plans, debug)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(
    env: Any,
    model: CustomModel,
    task_oracle: Any,
    subtask: str,
    val_annotations: Dict[str, Any],
    plans: defaultdict,
    debug: bool,
) -> bool:
    """
    Run the actual rollout on one subtask (which is one natural language instruction).

    Args:
        env: The CALVIN environment
        model: The model to evaluate
        task_oracle: Task oracle for checking task completion
        subtask: Single language instruction to execute
        val_annotations: Validation annotations for language goals
        plans: Dictionary to collect plan data for TSNE visualization
        debug: Whether to print debug information and visualize

    Returns:
        True if the subtask was completed successfully, False otherwise
    """
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()

    for step in range(EP_LEN):
        action = model.step(obs, lang_annotation)
        obs, _, _, current_info = env.step(action)
        if debug:
            img = env.render(mode="rgb_array")
            join_vis_lang(img, lang_annotation)
            # time.sleep(0.1)
        if step == 0:
            # for tsne plot, only if available
            collect_plan(model, plans, subtask)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
            return True
    if debug:
        print(colored("fail", "red"), end=" ")
    return False


class MCILModel(CustomModel):
    """
    Multi-modal Conditional Imitation Learning (MCIL) model for CALVIN evaluation.

    This model implements the CustomModel interface and provides functionality for
    evaluating trained models on the CALVIN benchmark with language instructions.
    """

    def __init__(self, config: Dict[str, Any], trainer: Any) -> None:
        """
        Initialize the MCIL model.

        Args:
            config: Configuration dictionary containing model and training parameters
            trainer: Trainer object that provides the act method
        """
        self.trainer = trainer
        self.random_robot_state = config["data"]["evaluation"]["random_robot_state"]
        self.device = config["device"]
        self.depth = False
        self.lang_clip = False
        self.config = config
        self.model, self.transform = clip.load(
            name=config["trainer"]["vlm"]["name"],
            device=config["device"],
            download_root=config["trainer"]["vlm"]["download_root"],
        )
        self.data_keys = {"rgb_static", "robot_obs", "rgb_gripper"}
        self.action_dim = config["trainer"]["model"]["action_dim"]
        self.relative_action = True
        self.logging_directory = os.path.join(
            config["logging"]["model_directory"],
            config["trainer"]["load"]["experiment_name"],
            config["trainer"]["load"]["run_id"],
        )

        if not os.path.exists(self.logging_directory):
            os.makedirs(self.logging_directory)

        self._load_lang_embeddings()
        self.record_video = None
        self.epoch = 0
        self.n_videos = 0
        self.past_seqs = []
        self.obs_seq = []

    def clean_past_seq(self) -> None:
        """
        Clear the past sequences stored in the model.
        """
        self.past_seqs = []

    def reset(self) -> None:
        """
        Reset the model's internal state for a new episode.

        This method is called at the beginning of each episode or sequence.
        """
        self.current_step = 0
        self.current_goal = None
        if len(self.obs_seq) > 0:
            self.past_seqs.append(self.obs_seq)
        self.obs_seq = []
        self.last_action = torch.zeros(self.action_dim).reshape(1, -1).to(self.device)

    def step(self, obs: Dict[str, Any], goal: str) -> Dict[str, Any]:
        """
        Take a step in the environment based on current observations and goal.

        Args:
            obs: Environment observations containing RGB images, robot state, etc.
            goal: Language instruction describing the desired task

        Returns:
            Dictionary containing the predicted action and action type
        """
        obs = self._unpack(obs)
        self._to_device(obs)

        # Create observations sequence
        if goal != self.current_goal:
            resample = True
        else:
            resample = False
        self.current_goal = goal
        self.current_instruction = self.encode_language_goal(goal)
        obs["instructions"] = self.current_instruction
        obs["actions"] = self.last_action
        obs["nl_inst"] = goal
        self.obs_seq.append(obs)
        seq = self.create_seq()

        # Act
        action = self.trainer.act(seq, self.device, resample=resample)
        self.last_action = action

        return self.postprocess_action(action)

    def postprocess_action(self, action: torch.Tensor) -> Dict[str, Any]:
        """
        Post-process the raw action tensor into the required format.

        Args:
            action: Raw action tensor from the model

        Returns:
            Dictionary containing the processed action and action type
        """
        # Convert to numpy
        if not self.relative_action:
            action = action.squeeze().cpu().detach().numpy()
            action = np.split(action, [3, 6])
            action[-1] = 1 if action[-1] > 0 else -1
            return {"action": action, "type": "cartesian_abs"}
        else:
            action = action.squeeze().cpu().detach().numpy()
            action[-1] = 1 if action[-1] > 0 else -1
            return {"action": action, "type": "cartesian_rel"}

    def encode_language_goal(self, goal: str) -> torch.Tensor:
        """
        Encode a language goal into a tensor representation.

        Args:
            goal: Language instruction string

        Returns:
            Tensor representation of the language goal
        """
        if self.lang_clip:
            return self.model.encode_text(clip.tokenize(goal).to(self.device))
        else:
            return torch.from_numpy(self.lang_embeddings[goal]).float().reshape(1, -1)

    def save_video_recording(self) -> None:
        """
        Save the current episode as a video recording with associated text.

        This method creates a video file from the observation sequence and logs it
        to wandb along with the corresponding language instructions.
        """
        # Add current observation sequence
        self.past_seqs.append(self.obs_seq)

        arr_obs = []
        nl_inst = []
        timestep = 0
        for seq in self.past_seqs:
            for obs in seq:
                arr_obs.append(obs["rgb_static"])
                nl_inst.append(f"{timestep}: {obs['nl_inst']}")

        video = torch.stack(arr_obs, dim=0).detach().cpu().numpy()
        text = "\n".join(nl_inst)

        log_dir = os.path.join(self.logging_directory, str(self.epoch))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Save video
        video_file = os.path.join(log_dir, f"{self.n_videos}.mp4")
        with open(video_file, "wb+") as file:
            write_video(video_file, video, fps=20)
        wandb.log({"evaluation/video": wandb.Video(video_file)})

        # Save text
        text_file = os.path.join(log_dir, f"{self.n_videos}.txt")
        with open(text_file, "w+") as file:
            file.write(text)

        self.n_videos += 1
        self.past_seqs = []
        self.obs_seq = []

    def create_seq(self) -> Dict[str, Any]:
        """
        Create a sequence tensor from the current observation sequence.

        Returns:
            Dictionary containing batched observation tensors and metadata
        """
        img_obs = []
        gripper_obs = []
        proprioception = []
        actions = []
        inst_length = []
        obs_length = []

        for obs in self.obs_seq:
            if self.depth:
                i_obs = torch.cat([obs["rgb_static"], obs["depth_static"].unsqueeze(-1)], dim=-1)
                g_obs = torch.cat([obs["rgb_gripper"], obs["depth_gripper"].unsqueeze(-1)], dim=-1)
            else:
                i_obs = obs["rgb_static"]
                g_obs = obs["rgb_gripper"]

            img_obs.append(i_obs)
            gripper_obs.append(g_obs)
            proprioception.append(obs["robot_obs"])
            actions.append(obs["actions"])

        # Concatenate observations
        img_obs = torch.stack(img_obs, dim=0)
        proprioception = torch.stack(proprioception, dim=0)
        gripper_obs = torch.stack(gripper_obs, dim=0)
        actions = torch.cat(actions, dim=0)

        # Form a batch
        gripper_obs = pad_sequence([gripper_obs], batch_first=True).float()
        img_obs = pad_sequence([img_obs], batch_first=True).float()
        proprioception = pad_sequence([proprioception], batch_first=True).float()
        actions = pad_sequence([actions], batch_first=True).float()
        instructions = pad_sequence([self.obs_seq[-1]["instructions"]], batch_first=True).float()
        inst_length = torch.tensor([1], dtype=torch.long).reshape(1, -1)
        obs_length = torch.tensor([len(self.obs_seq)], dtype=torch.long)
        mask = torch.ones(1).bool()

        return {
            "img_obs": img_obs,
            "gripper_obs": gripper_obs,
            "proprioceptions": proprioception,
            "actions": actions,
            "instructions": instructions,
            "inst_length": inst_length,
            "obs_length": obs_length,
            "mask": mask,
        }

    def _unpack(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unpack nested observation data into a flat dictionary.

        Args:
            data: Nested observation dictionary

        Returns:
            Flattened dictionary containing only the required data keys
        """
        new_data = {}
        for key, value in data.items():
            if isinstance(value, dict):
                for key2, value2 in value.items():
                    if key2 in self.data_keys:
                        new_data[key2] = value2
            else:
                if key in self.data_keys:
                    new_data[key] = value
        return new_data

    def _to_device(self, data: Dict[str, Any]) -> None:
        """
        Move numpy arrays in the data dictionary to the specified device.

        Args:
            data: Dictionary containing observation data
        """
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = torch.from_numpy(value).to(self.device)

    def _load_lang_embeddings(self) -> None:
        """
        Load language embeddings from the dataset.

        This method loads pre-computed language embeddings for the validation set
        and optionally maps instructions to task embeddings if task_embeddings is enabled.
        """
        filename = os.path.join(
            self.config["data"]["dataset_directory"], "validation", "lang_annotations", "embeddings.npy"
        )
        with open(filename, "rb") as file:
            embeddings = np.load(file, allow_pickle=True).item()

        if self.config["data"]["task_embeddings"]:
            self.inst2task = {value["ann"][0]: key for key, value in embeddings.items()}

            directory = os.path.join(self.config["data"]["dataset_directory"], "training")
            with open(os.path.join(directory, "row2task.pkl"), "rb") as file:
                row2task = pickle.load(file)
                task2row = {task: row for row, task in row2task.items()}

            with open(os.path.join(directory, "task2instructions.pkl"), "rb") as file:
                task2instructions = pickle.load(file)
                instructions2task = {}
                for task, instructions in task2instructions.items():
                    for inst in instructions:
                        instructions2task[inst] = task

            with open(os.path.join(directory, "lang_annotations", "auto_lang_ann.npy"), "rb") as file:
                lang = np.load(file, allow_pickle=True).item()
                task_embeddings = lang["language"]["task_emb"]

            self.lang_embeddings = {}
            for inst, task in self.inst2task.items():
                row = task2row[task]
                self.lang_embeddings[inst] = task_embeddings[row]
        else:
            self.lang_embeddings = {value["ann"][0]: value["emb"] for value in embeddings.values()}


def set_up_eval_config(config: Dict[str, Any], trainer: Any) -> Dict[str, Any]:
    """
    Set up the evaluation configuration for CALVIN evaluation.

    Args:
        config: Configuration dictionary containing model and dataset parameters
        trainer: Trainer object for the model

    Returns:
        Dictionary containing all components needed for evaluation
    """
    env = make_env(config["data"]["dataset_directory"])
    model = MCILModel(config, trainer)
    conf_dir = Path(__file__).absolute().parents[2] / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = os.path.join(
        config["logging"]["model_directory"],
        config["trainer"]["load"]["experiment_name"],
        config["trainer"]["load"]["run_id"],
    )

    eval_sequences = get_sequences(config["data"]["evaluation"]["num_eval_sequences"])

    return {
        "eval_sequences": eval_sequences,
        "model": model,
        "env": env,
        "eval_sequences": eval_sequences,
        "task_oracle": task_oracle,
        "val_annotations": val_annotations,
        "eval_log_dir": eval_log_dir,
        "n_videos_to_record": config["data"]["evaluation"]["n_videos_to_record"],
    }


def evaluate_calvin(eval_config: Dict[str, Any]) -> None:
    """
    Evaluate a CALVIN model using the provided configuration.

    Args:
        eval_config: Dictionary containing all evaluation components and parameters
    """
    evaluate_policy(
        eval_config["model"],
        eval_config["env"],
        eval_config["eval_sequences"],
        eval_config["task_oracle"],
        eval_config["val_annotations"],
        eval_config["eval_log_dir"],
        eval_config["n_videos_to_record"],
        debug=False,
    )


def main() -> None:
    """
    Main function for running CALVIN model evaluation.

    This function sets up argument parsing, loads checkpoints, and runs evaluation
    on the specified model(s) with the given configuration.
    """
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset root directory.")

    # arguments for loading default model
    parser.add_argument(
        "--train_folder", type=str, help="If calvin_agent was used to train, specify path to the log dir."
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help="Comma separated list of epochs for which checkpoints will be loaded",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path of the checkpoint",
    )
    parser.add_argument(
        "--last_k_checkpoints",
        type=int,
        help="Specify the number of checkpoints you want to evaluate (starting from last). Only used for calvin_agent.",
    )

    # arguments for loading custom model or custom language embeddings
    parser.add_argument(
        "--custom_model", action="store_true", help="Use this option to evaluate a custom model architecture."
    )

    parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment.")

    parser.add_argument("--eval_log_dir", default=None, type=str, help="Where to log the evaluation results.")

    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    args = parser.parse_args()

    # evaluate a custom model
    if args.custom_model:
        model = CustomModel()
        env = make_env(args.dataset_path)
        evaluate_policy(model, env, debug=args.debug)
    else:
        assert "train_folder" in args

        checkpoints = []
        if args.checkpoints is None and args.last_k_checkpoints is None and args.checkpoint is None:
            print("Evaluating model with last checkpoint.")
            checkpoints = [get_last_checkpoint(Path(args.train_folder))]
        elif args.checkpoints is not None:
            print(f"Evaluating model with checkpoints {args.checkpoints}.")
            checkpoints = get_checkpoints_for_epochs(Path(args.train_folder), args.checkpoints)
        elif args.checkpoints is None and args.last_k_checkpoints is not None:
            print(f"Evaluating model with last {args.last_k_checkpoints} checkpoints.")
            checkpoints = get_all_checkpoints(Path(args.train_folder))[-args.last_k_checkpoints :]
        elif args.checkpoint is not None:
            checkpoints = [Path(args.checkpoint)]

        env = None
        for checkpoint in checkpoints:
            epoch = get_epoch(checkpoint)
            model, env, _ = get_default_model_and_env(
                args.train_folder,
                args.dataset_path,
                checkpoint,
                env=env,
                device_id=args.device,
            )
            evaluate_policy(model, env, epoch, eval_log_dir=args.eval_log_dir, debug=args.debug, create_plan_tsne=True)


if __name__ == "__main__":
    main()
