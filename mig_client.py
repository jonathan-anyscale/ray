import pynvml
from typing import Dict, TypedDict, List
from pdb import set_trace as bp

class GPUInstanceResource(TypedDict):
    computeSlice: int
    memorySizeMB: int
    ofaCount: int

class GPUInstanceResourceInstance:
    def __init__(self, computeSlice: int = 0, memorySizeMB: int = 0, ofaCount: int = 0):
        self['computeSlice'] = computeSlice
        self['memorySizeMB'] = memorySizeMB
        self['ofaCount'] = ofaCount

class GPUInstanceConfig(TypedDict):
    name: str
    profile_id: int
    resource: GPUInstanceResource

class MIGManager:
    def __init__(self):
        # Initialize the NVML library
        pynvml.nvmlInit()

        # detect all mig-enabled gpu
        self.mig_enabled_gpus = []
        self.gpu_instance_config: Dict[int, List[GPUInstanceConfig]] = {}
        self.gpu_instance_resources: Dict[int, GPUInstanceResource] = {}
        num_gpus = pynvml.nvmlDeviceGetCount()
        for gpu_index in range(num_gpus):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                mig_mode, pending_mode = pynvml.nvmlDeviceGetMigMode(handle)
                if mig_mode == pynvml.NVML_DEVICE_MIG_ENABLE and pending_mode == mig_mode:
                    self.mig_enabled_gpus.append(gpu_index)
                    self.gpu_instance_info[gpu_index] = []
            except pynvml.NVMLError as err:
                print(f"Error checking GPU {gpu_index}: {err}")
        
        # get gpu mig resources (max compute, memory and ofa)

        # get all profile infos
        for mig_gpu in self.mig_enabled_gpus:
            handle = pynvml.nvmlDeviceGetHandleByIndex(mig_gpu)
            while True:
                try:
                    gi_info = pynvml.nvmlDeviceGetGpuInstanceProfileInfo(handle, profile_idx)
                    gi_resource = GPUInstanceResource(
                        gi_info.sliceCount,
                        gi_info.memorySizeMB,
                        gi_info.ofaCount
                    )
                    self.gpu_instance_info[mig_gpu].append({
                        "name": gi_info.name,
                        "profile_id": gi_info.id,
                        "resource": gi_resource
                    })
                    profile_idx += 1
                except pynvml.nvml.NVMLError:
                    break
        self._update_gpu_info()
    
    def __del__(self):
        # Shutdown the NVML library when the object is destroyed
        pynvml.nvmlShutdown()
    
    def get_gpu_mig_resources(gpu_index):
        if mig_index not in self.gpu_instance_info:
            return GPUInstanceResourceInstance()
        return self.gpu_instance_info[gpu_index]
    
    def create_mig_instance(self, gpu_index, mig_device_id):
        try:
            # Open the GPU
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            
            # Get MIG device information
            mig_info = pynvml.nvmlDeviceGetMigDeviceHandleByIndex(handle, mig_device_id)
            
            # Perform further configuration and setup as needed
            
            return f"MIG instance {mig_device_id} created on GPU {gpu_index}"
        except pynvml.NVMLError as err:
            return f"Error creating MIG instance: {err}"
    
    def delete_mig_instance(self, gpu_index, mig_device_id):
        try:
            # Open the GPU
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            
            # Get MIG device information
            mig_info = pynvml.nvmlDeviceGetMigDeviceHandleByIndex(handle, mig_device_id)
            
            # Perform cleanup and deletion
            
            return f"MIG instance {mig_device_id} deleted on GPU {gpu_index}"
        except pynvml.NVMLError as err:
            return f"Error deleting MIG instance: {err}"

if __name__ == "__main__":
    mig_manager = MIGManager()
    
    # Example: Create a MIG instance on GPU 0 with MIG device ID 0
    create_result = mig_manager.create_mig_instance(0, 0)
    print(create_result)
    
    # Example: Delete a MIG instance on GPU 0 with MIG device ID 0
    delete_result = mig_manager.delete_mig_instance(0, 0)
    print(delete_result)
