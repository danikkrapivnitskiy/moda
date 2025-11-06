"""
GPU Compatibility Checker

Enterprise-grade GPU compatibility validation for MoDA pipeline.
Provides centralized GPU capability checks, memory validation, and
batch size recommendations based on GPU architecture.

This module uses only existing dependencies (torch, standard library)
to avoid increasing Docker image size.
"""

from typing import Dict, Any, Optional, Tuple


class GPUCompatibilityChecker:
    """
    GPU compatibility checker for MoDA pipeline.
    
    Provides methods to check GPU compute capability, memory,
    and validate batch sizes for different GPU architectures.
    """
    
    def __init__(self, device_id: int = 0):
        """
        Initialize GPU compatibility checker.
        
        Args:
            device_id: CUDA device ID to check
        """
        self.device_id = device_id
        self._gpu_props = None
        self._memory_info = None
    
    def _get_gpu_props(self):
        """Get GPU properties (cached)"""
        if self._gpu_props is None:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.set_device(self.device_id)
                    self._gpu_props = torch.cuda.get_device_properties(self.device_id)
                else:
                    self._gpu_props = None
            except Exception:
                self._gpu_props = None
        return self._gpu_props
    
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def check_compute_capability(self) -> Optional[Tuple[int, int]]:
        """
        Check GPU compute capability.
        
        Returns:
            Tuple of (major, minor) compute capability, or None if unavailable
        """
        props = self._get_gpu_props()
        if props is None:
            return None
        return (props.major, props.minor)
    
    def check_fp16_support(self) -> bool:
        """
        Check if GPU supports FP16 (half precision).
        
        FP16 requires compute capability >= 7.0 (Volta architecture and newer).
        
        Returns:
            True if FP16 is supported, False otherwise
        """
        compute_cap = self.check_compute_capability()
        if compute_cap is None:
            return False
        major, _ = compute_cap
        # Compute capability 7.0+ (Volta, Turing, Ampere, Ada, Hopper)
        return major >= 7
    
    def check_torch_compile_support(self) -> bool:
        """
        Check if GPU supports torch.compile optimization.
        
        torch.compile with max-autotune requires compute capability >= 7.0.
        
        Returns:
            True if torch.compile is supported, False otherwise
        """
        compute_cap = self.check_compute_capability()
        if compute_cap is None:
            return False
        major, _ = compute_cap
        # Compute capability 7.0+ required for torch.compile optimizations
        return major >= 7
    
    def get_gpu_memory_info(self) -> Dict[str, Any]:
        """
        Get GPU memory information.
        
        Returns:
            Dictionary with memory information:
            - total_gb: Total GPU memory in GB
            - allocated_gb: Currently allocated memory in GB
            - reserved_gb: Currently reserved memory in GB
            - free_gb: Free memory in GB
            - available: Whether GPU is available
        """
        if not self.is_cuda_available():
            return {
                "available": False,
                "total_gb": 0,
                "allocated_gb": 0,
                "reserved_gb": 0,
                "free_gb": 0
            }
        
        try:
            import torch
            props = self._get_gpu_props()
            if props is None:
                return {
                    "available": False,
                    "total_gb": 0,
                    "allocated_gb": 0,
                    "reserved_gb": 0,
                    "free_gb": 0
                }
            
            torch.cuda.set_device(self.device_id)
            total_gb = props.total_memory / (1024**3)
            allocated_gb = torch.cuda.memory_allocated(self.device_id) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(self.device_id) / (1024**3)
            free_gb = total_gb - reserved_gb
            
            return {
                "available": True,
                "total_gb": total_gb,
                "allocated_gb": allocated_gb,
                "reserved_gb": reserved_gb,
                "free_gb": free_gb
            }
        except Exception as e:
            print(f"[WARN] Failed to get GPU memory info: {e}")
            return {
                "available": False,
                "total_gb": 0,
                "allocated_gb": 0,
                "reserved_gb": 0,
                "free_gb": 0
            }
    
    def get_recommended_batch_size(self, batch_type: str = "inference") -> int:
        """
        Get recommended batch size based on GPU memory.
        
        Args:
            batch_type: Type of batch operation ("inference", "warp_decode")
        
        Returns:
            Recommended batch size
        """
        memory_info = self.get_gpu_memory_info()
        if not memory_info["available"]:
            # Conservative default if GPU not available
            return 10 if batch_type == "warp_decode" else 20
        
        total_gb = memory_info["total_gb"]
        
        # Recommended batch sizes based on GPU memory
        # Conservative limits to prevent OOM
        if batch_type == "warp_decode":
            # warp_decode is memory-intensive, needs more conservative limits
            if total_gb < 30:  # RTX 4090, RTX 3090 (24GB)
                return 25
            elif total_gb < 45:  # A6000, A40 (40-48GB)
                return 75
            else:  # A100, H100 (80GB+)
                return 100
        else:
            # Regular inference batch size
            if total_gb < 30:  # RTX 4090, RTX 3090 (24GB)
                return 20
            elif total_gb < 45:  # A6000, A40 (40-48GB)
                return 50
            else:  # A100, H100 (80GB+)
                return 100
    
    def validate_batch_size_for_gpu(
        self,
        batch_size: int,
        batch_type: str = "inference"
    ) -> Dict[str, Any]:
        """
        Validate batch size against GPU memory capabilities.
        
        Args:
            batch_size: Batch size to validate
            batch_type: Type of batch operation ("inference", "warp_decode")
        
        Returns:
            Dictionary with validation results:
            - valid: Whether batch size is valid
            - warnings: List of warning messages
            - errors: List of error messages
            - recommended: Recommended batch size
            - max_safe: Maximum safe batch size for this GPU
        """
        warnings = []
        errors = []
        
        if not self.is_cuda_available():
            warnings.append("CUDA not available, using conservative batch size limits")
            max_safe = 20 if batch_type == "warp_decode" else 10
            if batch_size > max_safe:
                errors.append(
                    f"batch_size {batch_size} exceeds safe limit {max_safe} "
                    f"for non-GPU environment"
                )
                return {
                    "valid": False,
                    "warnings": warnings,
                    "errors": errors,
                    "recommended": max_safe,
                    "max_safe": max_safe
                }
        
        memory_info = self.get_gpu_memory_info()
        if not memory_info["available"]:
            warnings.append("GPU memory info unavailable, using conservative limits")
            max_safe = 25 if batch_type == "warp_decode" else 20
            if batch_size > max_safe:
                errors.append(
                    f"batch_size {batch_size} exceeds safe limit {max_safe} "
                    f"(GPU memory info unavailable)"
                )
            return {
                "valid": len(errors) == 0,
                "warnings": warnings,
                "errors": errors,
                "recommended": max_safe,
                "max_safe": max_safe
            }
        
        total_gb = memory_info["total_gb"]
        recommended = self.get_recommended_batch_size(batch_type)
        
        # Determine max safe batch size based on GPU memory
        if batch_type == "warp_decode":
            if total_gb < 30:
                max_safe = 25
            elif total_gb < 45:
                max_safe = 75
            else:
                max_safe = 100
        else:
            if total_gb < 30:
                max_safe = 20
            elif total_gb < 45:
                max_safe = 50
            else:
                max_safe = 100
        
        # Validate batch size
        if batch_size > max_safe:
            errors.append(
                f"batch_size {batch_size} exceeds safe limit {max_safe} "
                f"for {total_gb:.1f}GB GPU"
            )
        elif batch_size > max_safe * 0.8:
            warnings.append(
                f"batch_size {batch_size} is close to safe limit {max_safe} "
                f"for {total_gb:.1f}GB GPU"
            )
        
        # Hard limit check (absolute maximum)
        absolute_max = 100
        if batch_size > absolute_max:
            errors.append(
                f"batch_size {batch_size} exceeds absolute maximum {absolute_max}"
            )
        
        return {
            "valid": len(errors) == 0,
            "warnings": warnings,
            "errors": errors,
            "recommended": recommended,
            "max_safe": max_safe,
            "gpu_memory_gb": total_gb
        }
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Get comprehensive GPU information.
        
        Returns:
            Dictionary with GPU information:
            - name: GPU name
            - compute_capability: Compute capability (major.minor)
            - total_memory_gb: Total memory in GB
            - fp16_supported: Whether FP16 is supported
            - torch_compile_supported: Whether torch.compile is supported
            - available: Whether GPU is available
        """
        if not self.is_cuda_available():
            return {
                "available": False,
                "name": "N/A",
                "compute_capability": None,
                "total_memory_gb": 0,
                "fp16_supported": False,
                "torch_compile_supported": False
            }
        
        props = self._get_gpu_props()
        if props is None:
            return {
                "available": False,
                "name": "N/A",
                "compute_capability": None,
                "total_memory_gb": 0,
                "fp16_supported": False,
                "torch_compile_supported": False
            }
        
        compute_cap = self.check_compute_capability()
        compute_cap_str = f"{compute_cap[0]}.{compute_cap[1]}" if compute_cap else "N/A"
        
        return {
            "available": True,
            "name": props.name,
            "compute_capability": compute_cap_str,
            "total_memory_gb": props.total_memory / (1024**3),
            "fp16_supported": self.check_fp16_support(),
            "torch_compile_supported": self.check_torch_compile_support()
        }

