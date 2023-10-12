import os
import sys
import logging
import asyncio
import subprocess
import copy
from pathlib import Path
from typing import Tuple, List, Dict, Optional

from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.plugin import RuntimeEnvPlugin
from ray._private.utils import (
    try_to_create_directory,
)
from ray.exceptions import RuntimeEnvSetupError

default_logger = logging.getLogger(__name__)

# Nsight options used when runtime_env={"_nsight": "default"}
NSIGHT_DEFAULT_CONFIG = {
    "-t": "cuda,cudnn,cublas,nvtx",
    "-o": "'worker_process_%p'",
    "--cudabacktrace": "all",
    "--stop-on-exit": "true",
}

def parse_nsight_config(nsight_config: Dict[str, str]) -> List[str]:
    """
    Function to convert dictionary of nsight options into
    nsight command line

    Args:
        nsight_config: dictionary mapping nsight option to it's value
    Returns:
        a list of string consists of nsys profile command line and options
    """
    nsight_cmd = ["nsys", "profile"]
    for option, option_val in nsight_config.items():
        if option[:2] == "--":
            nsight_cmd.append(f"{option}={option_val}")
        elif option[0] == "-":
            nsight_cmd += [option, option_val]
    return nsight_cmd


class NsightPlugin(RuntimeEnvPlugin):
    name = "_nsight"

    def __init__(self, resources_dir: str):
        self.nsight_cmd = []

        # replace this with better way to get logs dir
        session_dir, runtime_dir = os.path.split(resources_dir)
        self._profilers_dir = Path(session_dir) / "logs" / "nsight"
        try_to_create_directory(self._profilers_dir)
    
    async def _check_nsight_script(self, nsight_config: Dict[str, str]) -> Tuple[bool, str]:
        """
        Function to validate if nsight_config is a valid nsight profile options
        Args:
            nsight_config: dictionary mapping nsight option to it's value
        Returns:
            a tuple consists of a boolean indicating if the nsight_config 
            is valid option and an error message if the nsight_config is invalid
        """

        # use empty as nsight report test filename
        nsight_config_copy = copy.deepcopy(nsight_config)
        nsight_config_copy["-o"] = str(Path(self._profilers_dir) / "empty")
        nsight_cmd = parse_nsight_config(nsight_config_copy)
        try:
            nsight_cmd = nsight_cmd + ["python", "-c", '""']
            process = await asyncio.create_subprocess_exec(
                *nsight_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            clean_up_cmd = ["rm", f"{nsight_config_copy['-o']}.nsys-rep"]
            if process.returncode == 0:
                cleanup_process = await asyncio.create_subprocess_exec(
                    *clean_up_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                _, _ = await cleanup_process.communicate()
                return True, None
            else:
                cleanup_process = await asyncio.create_subprocess_exec(
                    *clean_up_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                _, _ = await cleanup_process.communicate()
                error_msg = (
                    stderr.strip()
                    if stderr.strip() != ""
                    else stdout.strip()
                )
                return False, error_msg
        except FileNotFoundError:
            return False, ("nsight is not installed")

    async def create(
        self,
        uri: Optional[str],
        runtime_env: "RuntimeEnv",  # noqa: F821
        context: RuntimeEnvContext,
        logger: logging.Logger = default_logger,
    ) -> int:
        nsight_config = runtime_env.nsight()
        if not nsight_config:
            return 0

        if nsight_config and sys.platform != "linux":
            raise RuntimeEnvSetupError(
                "Nsight CLI is only available in Linux.\n"
                "More information can be found in "
                "https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html"
            )

        if isinstance(nsight_config, str):
            if nsight_config == "default":
                nsight_config = NSIGHT_DEFAULT_CONFIG
            else:
                raise RuntimeEnvSetupError(
                    f"Unsupported nsight config: {nsight_config}. "
                    "The supported config is 'default' or "
                    "Dictionary of nsight options"
                )

        is_valid_nsight_cmd, error_msg = await self._check_nsight_script(nsight_config)
        if not is_valid_nsight_cmd:
            logger.warning(error_msg)
            raise RuntimeEnvSetupError(
                "nsight profile failed to run with the following "
                f"error message:\n {error_msg}"
            )
        # add set output path to logs dir
        nsight_config["-o"] = str(Path(self._profilers_dir) / nsight_config.get(
            "-o", NSIGHT_DEFAULT_CONFIG["-o"]
        ))

        self.nsight_cmd = parse_nsight_config(nsight_config)
        return 0

    def modify_context(
        self,
        uris: List[str],
        runtime_env: "RuntimeEnv",  # noqa: F821
        context: RuntimeEnvContext,
        logger: Optional[logging.Logger] = default_logger,
    ):
        logger.info("Running nsight profiler")
        context.py_executable = " ".join(self.nsight_cmd) + " python"
