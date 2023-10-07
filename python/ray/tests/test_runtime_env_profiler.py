import os
import glob
import sys

import pytest

import ray
from ray._private.runtime_env.nsight import parse_nsight_config
from ray._private.test_utils import wait_for_condition
from ray.exceptions import RuntimeEnvSetupError

class TestNsightProfiler:

    @pytest.mark.skipif(
        os.environ.get("CI") and 
        sys.platform != "linux",
        reason="Requires PR wheels built in CI, so only run on linux CI machines.",
    )
    def test_nsight_basic(
        self,
        shutdown_only,
    ):
        """Test Nsight profile generate report on logs dir"""
        ray.init()
        session_dir = ray.worker._global_node.get_session_dir_path()
        logs_dir = f"{session_dir}/logs/nsight"

        # test nsight default config
        @ray.remote(runtime_env={ "nsight": "default" })
        def test_generate_report():
            return 0
        
        ray.get(test_generate_report.remote())

        wait_for_condition(lambda: len(os.listdir(logs_dir)) == 1)
        nsys_reports = glob.glob(os.path.join(f"{logs_dir}/*.nsys-rep"))
        assert len(nsys_reports) == 1
        for report in nsys_reports:
            assert "worker_process_" in report
            os.remove(report)
        
        # wait cleanup to finish
        wait_for_condition(lambda: len(os.listdir(logs_dir)) == 0)
        
        # test nsight custom filename
        CUSTOM_REPORT = "custom_report"
        @ray.remote(runtime_env={ "nsight": {
            "-o": CUSTOM_REPORT
        } })
        def test_generate_custom_report():
            return 0
        
        ray.get(test_generate_custom_report.remote())
        wait_for_condition(lambda: len(os.listdir(logs_dir)) == 1)
        nsys_reports = glob.glob(os.path.join(f"{logs_dir}/*.nsys-rep"))
        assert len(nsys_reports) == 1
        for report in nsys_reports:
            assert f"{CUSTOM_REPORT}.nsys-rep" in report
            os.remove(report)

    @pytest.mark.skipif(
        #os.environ.get("CI") and 
        sys.platform != "linux",
        reason="Requires PR wheels built in CI, so only run on linux CI machines.",
    )
    def test_nsight_multiple_tasks(
        self,
        shutdown_only,
    ):
        """Test Nsight profile on multiple ray tasks/actors"""
        ray.init()
        session_dir = ray.worker._global_node.get_session_dir_path()
        logs_dir = f"{session_dir}/logs/nsight"

        # test nsight default config
        @ray.remote(runtime_env={ "nsight": "default" })
        def test_generate_report():
            return 0
        
        ray.get([test_generate_report.remote(), test_generate_report.remote()])
        wait_for_condition(lambda: len(os.listdir(logs_dir)) == 2)
        nsys_reports = glob.glob(os.path.join(f"{logs_dir}/*.nsys-rep"))
        assert len(nsys_reports) == 2
        for report in nsys_reports:
            assert "worker_process_" in report
            os.remove(report)

    @pytest.mark.skipif(
        os.environ.get("CI") and 
        sys.platform != "linux",
        reason="Requires PR wheels built in CI, so only run on linux CI machines.",
    )
    def test_nsight_invalid_option(
        self,
        shutdown_only,
    ):
        """Test Nsight profile for invalid option"""
        ray.init()
        session_dir = ray.worker._global_node.get_session_dir_path()
        logs_dir = f"{session_dir}/logs/nsight/"

        # test nsight default config
        @ray.remote(runtime_env={ "nsight": {
            "-not-option": "random",
        } })
        def test_invalid_nsight():
            return 0

        with pytest.raises(RuntimeEnvSetupError) as option_error:
            ray.get(test_invalid_nsight.remote())

        nsys_reports = glob.glob(os.path.join(f"{logs_dir}/*.nsys-rep"))
        assert len(nsys_reports) == 0

        """Test Nsight profile for unavailable string option"""

        # test nsight default config
        @ray.remote(runtime_env={ "nsight": "not_default" })
        def test_wrong_config_nsight():
            return 0

        with pytest.raises(RuntimeEnvSetupError):
            ray.get(test_wrong_config_nsight.remote())

        nsys_reports = glob.glob(os.path.join(f"{logs_dir}/*.nsys-rep"))
        assert len(nsys_reports) == 0

    @pytest.mark.skipif(
        sys.platform == "linux",
        reason="For non linux system, Nsight CLI is not supported",
    )
    def test_nsight_unsupported(
        self,
        shutdown_only,
    ):
        """Test Nsight profile for unsupported OS"""
        ray.init()
        session_dir = ray.worker._global_node.get_session_dir_path()
        logs_dir = f"{session_dir}/logs/nsight/"

        # test nsight default config
        @ray.remote(runtime_env={ "nsight": "default"})
        def test_nsight_not_supported():
            return 0

        with pytest.raises(RuntimeEnvSetupError):
            ray.get(test_nsight_not_supported.remote())

        nsys_reports = glob.glob(os.path.join(f"{logs_dir}/*.nsys-rep"))
        assert len(nsys_reports) == 0

    def test_parse_nsight_config(self):
        """ Test parse nsight config into nsight command prefix """
        nsight_config = {
            "-one-dash": "no=",
            "--two-dash": "with=",
            "invalid_flag": "nothing_return"
        }
        nsight_cmd = parse_nsight_config(nsight_config)
        assert nsight_cmd == ["nsys", "profile", "-one-dash", "no=", "--two-dash=with="]


if __name__ == "__main__":
    if os.environ.get("PARALLEL_CI"):
        sys.exit(pytest.main(["-n", "auto", "--boxed", "-vs", __file__]))
    else:
        sys.exit(pytest.main(["-sv", __file__]))
