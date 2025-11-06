"""
Enterprise Configuration Manager

Centralized configuration management system for MoDA pipeline.
Provides single source of truth configuration with automatic merging,
environment variable overrides, and validation.

This module implements enterprise-grade configuration management:
- Single source of truth: runpod_config.yaml
- Automatic merging into nested configs
- Environment variable overrides
- Type and range validation
- Backward compatibility
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from omegaconf import OmegaConf, DictConfig

# Lazy import to avoid circular dependencies
# Per-device cache to support multiple GPU devices
_gpu_checkers = {}  # device_id -> GPUCompatibilityChecker

def _get_gpu_checker(device_id: int = 0):
    """Get GPU compatibility checker instance (lazy import, per-device cache)"""
    global _gpu_checkers
    if device_id not in _gpu_checkers:
        try:
            from utils.gpu_compatibility import GPUCompatibilityChecker
            _gpu_checkers[device_id] = GPUCompatibilityChecker(device_id)
        except ImportError:
            # Fallback if module not available
            return None
    return _gpu_checkers[device_id]


class ConfigManager:
    """
    Enterprise configuration manager for MoDA pipeline.
    
    Provides centralized configuration loading, merging, validation,
    and environment variable override support.
    """
    
    def __init__(self, primary_config_path: Optional[str] = None):
        """
        Initialize ConfigManager.
        
        Args:
            primary_config_path: Path to primary config file (runpod_config.yaml).
                                If None, uses default location.
        """
        if primary_config_path is None:
            # Default to runpod_config.yaml in project root
            # Try multiple possible locations for flexibility
            possible_paths = [
                Path(__file__).parent.parent.parent / "runpod_config.yaml",  # From src/utils/
                Path("/app/runpod_config.yaml"),  # Docker container
                Path("runpod_config.yaml"),  # Current directory
            ]
            
            # Find first existing path
            for path in possible_paths:
                if path.exists():
                    primary_config_path = path
                    break
            else:
                # Default to project root relative to this file
                primary_config_path = Path(__file__).parent.parent.parent / "runpod_config.yaml"
        else:
            primary_config_path = Path(primary_config_path)
        
        self.primary_config_path = Path(primary_config_path)
        self._primary_config: Optional[DictConfig] = None
        self._validation_errors: List[str] = []
    
    def load_primary_config(self) -> DictConfig:
        """
        Load primary configuration file with validation.
        
        Returns:
            DictConfig: Loaded primary configuration
            
        Raises:
            FileNotFoundError: If primary config file doesn't exist
            ValueError: If configuration is invalid
        """
        if not self.primary_config_path.exists():
            raise FileNotFoundError(
                f"Primary config file not found: {self.primary_config_path}"
            )
        
        try:
            self._primary_config = OmegaConf.load(self.primary_config_path)
            
            # Apply environment variable overrides
            self._apply_env_overrides(self._primary_config)
            
            # Validate configuration
            self._validate_primary_config(self._primary_config)
            
            if self._validation_errors:
                error_msg = "Configuration validation failed:\n" + "\n".join(
                    f"  - {err}" for err in self._validation_errors
                )
                raise ValueError(error_msg)
            
            # Apply GPU compatibility fixes (if CUDA available)
            # This is non-fatal - if GPU check fails, continue with config as-is
            try:
                device_id = self._primary_config.get('inference', {}).get('device_id', 0)
                self.apply_gpu_compatibility_fixes(self._primary_config, device_id)
            except Exception as e:
                print(f"[WARN] GPU compatibility check failed (non-fatal): {e}")
                print("[INFO] Continuing with configuration as-is")
            
            return self._primary_config
            
        except Exception as e:
            raise ValueError(f"Failed to load primary config: {e}") from e
    
    def merge_configs(
        self,
        primary_config: DictConfig,
        nested_config_path: str,
        section: Optional[str] = None
    ) -> DictConfig:
        """
        Merge primary config values into nested config.
        
        Args:
            primary_config: Primary configuration (from runpod_config.yaml)
            nested_config_path: Path to nested config file to merge into
            section: Section name in primary config to merge (e.g., 'inference', 'motion_processor', 'performance')
                     If None, merges all relevant sections
            
        Returns:
            DictConfig: Merged configuration
        """
        nested_path = Path(nested_config_path)
        
        # Resolve relative paths relative to primary config location or current directory
        if not nested_path.is_absolute():
            # Try relative to primary config directory first
            primary_dir = self.primary_config_path.parent
            possible_paths = [
                primary_dir / nested_path,
                Path.cwd() / nested_path,
                Path("/app") / nested_path,  # Docker container
            ]
            
            for path in possible_paths:
                if path.exists():
                    nested_path = path
                    break
            else:
                # If still not found, try the original path
                if not nested_path.exists():
                    raise FileNotFoundError(
                        f"Nested config file not found: {nested_config_path}\n"
                        f"Tried: {[str(p) for p in possible_paths]}"
                    )
        elif not nested_path.exists():
            raise FileNotFoundError(f"Nested config file not found: {nested_path}")
        
        # Load nested config
        nested_config = OmegaConf.load(nested_path)
        
        # Determine which sections to merge
        if section:
            sections_to_merge = [section]
        else:
            # Auto-detect sections based on config file name
            config_name = nested_path.stem
            if 'inference' in config_name:
                sections_to_merge = ['inference']
            elif 'liveportrait' in config_name or 'motion' in config_name:
                sections_to_merge = ['motion_processor', 'performance']  # Added performance
            else:
                sections_to_merge = []
        
        # Merge primary config sections into nested config
        # Nested configs have flat structure, so we merge section values directly
        # This adds all parameters from primary config, even if they don't exist in nested config
        for section_name in sections_to_merge:
            if section_name in primary_config:
                section_config = primary_config[section_name]
                
                # Special handling for performance section → liveportrait_config.yaml
                if section_name == 'performance' and ('liveportrait' in str(nested_path) or 'motion' in str(nested_path)):
                    # Map performance parameters to liveportrait config parameters
                    performance_mapping = {
                        'enable_torch_compile': 'flag_do_torch_compile',
                        'use_half_precision': 'flag_use_half_precision',
                        'max_video_length': 'max_video_length',  # Direct mapping (if not already set)
                    }
                    
                    for perf_key, liveportrait_key in performance_mapping.items():
                        if perf_key in section_config:
                            # For max_video_length, only set if not already set by inference section
                            if perf_key == 'max_video_length' and 'max_video_length' in nested_config:
                                # Skip if already set (inference section takes precedence)
                                continue
                            nested_config[liveportrait_key] = section_config[perf_key]
                            print(f"[INFO] Merged performance.{perf_key} → {liveportrait_key} = {section_config[perf_key]}")
                else:
                    # Standard merge for other sections
                    for key, value in section_config.items():
                        nested_config[key] = value
        
        # Also merge video_quality if present (flat merge for nested configs)
        # Video quality parameters are merged into both inference and motion_processor configs
        if 'video_quality' in primary_config:
            video_quality = primary_config.video_quality
            for key in ['crf', 'codec', 'preset', 'output_format', 'pixelformat']:
                if key in video_quality:
                    nested_config[key] = video_quality[key]
        
        return nested_config
    
    def _apply_env_overrides(self, config: DictConfig) -> None:
        """
        Apply environment variable overrides to configuration.
        
        Environment variable format: MODA_<SECTION>_<PARAM>
        Examples:
            MODA_INFERENCE_BATCH_SIZE=20
            MODA_MOTION_PROCESSOR_WARP_DECODE_BATCH_SIZE=30
            MODA_VIDEO_QUALITY_CRF=23
            MODA_PERFORMANCE_ENABLE_TORCH_COMPILE=True
            MODA_PERFORMANCE_USE_HALF_PRECISION=True
            MODA_MEMORY_OPTIMIZATION_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        
        Args:
            config: Configuration to apply overrides to (modified in place)
        """
        overrides_applied = []
        
        # Iterate through all environment variables with MODA_ prefix
        for env_key, env_value in os.environ.items():
            if not env_key.startswith('MODA_'):
                continue
            
            # Parse MODA_<SECTION>_<PARAM> format
            parts = env_key[5:].split('_')  # Remove 'MODA_' prefix
            if len(parts) < 2:
                continue
            
            section = parts[0].lower()
            param = '_'.join(parts[1:]).lower()
            
            # Convert value to appropriate type
            try:
                # Try to convert to int
                if env_value.isdigit() or (env_value.startswith('-') and env_value[1:].isdigit()):
                    typed_value = int(env_value)
                # Try to convert to float
                elif env_value.replace('.', '', 1).replace('-', '', 1).isdigit():
                    typed_value = float(env_value)
                # Try to convert to bool
                elif env_value.lower() in ('true', 'false'):
                    typed_value = env_value.lower() == 'true'
                else:
                    typed_value = env_value
                
                # Apply override
                if section in config:
                    if param in config[section]:
                        config[section][param] = typed_value
                        overrides_applied.append(f"{env_key}={env_value}")
                    elif section == 'memory_optimization' and param == 'pytorch_cuda_alloc_conf':
                        # Special case for memory optimization
                        if 'memory_optimization' not in config:
                            config['memory_optimization'] = {}
                        config['memory_optimization'][param] = typed_value
                        overrides_applied.append(f"{env_key}={env_value}")
                elif section == 'memory_optimization':
                    # Create memory_optimization section if it doesn't exist
                    if 'memory_optimization' not in config:
                        config['memory_optimization'] = {}
                    config['memory_optimization'][param] = typed_value
                    overrides_applied.append(f"{env_key}={env_value}")
                elif section == 'performance':
                    # Create performance section if it doesn't exist
                    if 'performance' not in config:
                        config['performance'] = {}
                    config['performance'][param] = typed_value
                    overrides_applied.append(f"{env_key}={env_value}")
                    
            except Exception as e:
                print(f"[WARN] Failed to apply environment override {env_key}={env_value}: {e}")
        
        if overrides_applied:
            print(f"[INFO] Applied {len(overrides_applied)} environment variable overrides:")
            for override in overrides_applied:
                print(f"  - {override}")
    
    def _validate_primary_config(self, config: DictConfig) -> None:
        """
        Validate primary configuration schema.
        
        Args:
            config: Configuration to validate
        """
        self._validation_errors = []
        
        # Validate inference section
        if 'inference' in config:
            inference = config.inference
            self._validate_int_range(inference, 'batch_size', min_val=1, max_val=1000, path='inference.batch_size')
            self._validate_int_range(inference, 'device_id', min_val=0, max_val=15, path='inference.device_id')
            self._validate_int_range(inference, 'source_max_dim', min_val=64, max_val=4096, path='inference.source_max_dim')
            self._validate_int_range(inference, 'input_height', min_val=64, max_val=1024, path='inference.input_height')
            self._validate_int_range(inference, 'input_width', min_val=64, max_val=1024, path='inference.input_width')
            self._validate_int_range(inference, 'output_fps', min_val=1, max_val=120, path='inference.output_fps')
            self._validate_int_range(inference, 'max_video_length', min_val=50, max_val=2000, path='inference.max_video_length')
            if 'warp_decode_batch_size' in inference:
                self._validate_int_range(inference, 'warp_decode_batch_size', min_val=1, max_val=200, path='inference.warp_decode_batch_size')
        
        # Validate motion_processor section
        if 'motion_processor' in config:
            motion = config.motion_processor
            self._validate_int_range(motion, 'batch_size', min_val=1, max_val=1000, path='motion_processor.batch_size')
            if 'warp_decode_batch_size' in motion:
                self._validate_int_range(motion, 'warp_decode_batch_size', min_val=1, max_val=200, path='motion_processor.warp_decode_batch_size')
            self._validate_int_range(motion, 'source_max_dim', min_val=64, max_val=4096, path='motion_processor.source_max_dim')
            self._validate_int_range(motion, 'input_height', min_val=64, max_val=1024, path='motion_processor.input_height')
            self._validate_int_range(motion, 'input_width', min_val=64, max_val=1024, path='motion_processor.input_width')
            self._validate_int_range(motion, 'output_height', min_val=64, max_val=4096, path='motion_processor.output_height')
            self._validate_int_range(motion, 'output_width', min_val=64, max_val=4096, path='motion_processor.output_width')
            self._validate_int_range(motion, 'output_fps', min_val=1, max_val=120, path='motion_processor.output_fps')
        
        # Validate performance section
        if 'performance' in config:
            performance = config.performance
            if 'enable_torch_compile' in performance:
                if not isinstance(performance.enable_torch_compile, bool):
                    self._validation_errors.append(
                        f"performance.enable_torch_compile must be a boolean, got {type(performance.enable_torch_compile).__name__}"
                    )
            if 'use_half_precision' in performance:
                if not isinstance(performance.use_half_precision, bool):
                    self._validation_errors.append(
                        f"performance.use_half_precision must be a boolean, got {type(performance.use_half_precision).__name__}"
                    )
            if 'max_video_length' in performance:
                self._validate_int_range(performance, 'max_video_length', min_val=50, max_val=2000, path='performance.max_video_length')
        
        # Validate video_quality section
        if 'video_quality' in config:
            video = config.video_quality
            self._validate_int_range(video, 'crf', min_val=0, max_val=51, path='video_quality.crf')
            if 'codec' in video:
                if not isinstance(video.codec, str):
                    self._validation_errors.append(f"video_quality.codec must be a string, got {type(video.codec)}")
                elif video.codec not in ['libx264', 'libx265', 'h264', 'hevc']:
                    self._validation_errors.append(f"video_quality.codec must be one of ['libx264', 'libx265', 'h264', 'hevc'], got {video.codec}")
            if 'preset' in video:
                if not isinstance(video.preset, str):
                    self._validation_errors.append(f"video_quality.preset must be a string, got {type(video.preset)}")
                elif video.preset not in ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow']:
                    self._validation_errors.append(f"video_quality.preset must be a valid FFmpeg preset, got {video.preset}")
        
        # Validate memory_optimization section
        if 'memory_optimization' in config:
            memory = config.memory_optimization
            if 'pytorch_cuda_alloc_conf' in memory:
                if not isinstance(memory.pytorch_cuda_alloc_conf, str):
                    self._validation_errors.append(f"memory_optimization.pytorch_cuda_alloc_conf must be a string, got {type(memory.pytorch_cuda_alloc_conf)}")
    
    def _validate_int_range(
        self,
        config: DictConfig,
        key: str,
        min_val: int,
        max_val: int,
        path: str
    ) -> None:
        """
        Validate integer value is within range.
        
        Args:
            config: Configuration section
            key: Key to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            path: Full path for error messages
        """
        if key not in config:
            return
        
        value = config[key]
        if not isinstance(value, int):
            self._validation_errors.append(
                f"{path} must be an integer, got {type(value).__name__}"
            )
        elif value < min_val or value > max_val:
            self._validation_errors.append(
                f"{path} must be between {min_val} and {max_val}, got {value}"
            )
    
    def get_memory_optimization_env(self, config: DictConfig) -> Dict[str, str]:
        """
        Get environment variables for memory optimization.
        
        Args:
            config: Primary configuration
            
        Returns:
            Dictionary of environment variable name -> value
        """
        env_vars = {}
        
        if 'memory_optimization' in config:
            memory = config.memory_optimization
            if 'pytorch_cuda_alloc_conf' in memory:
                env_vars['PYTORCH_CUDA_ALLOC_CONF'] = memory.pytorch_cuda_alloc_conf
        
        return env_vars
    
    def validate_gpu_compatibility(
        self,
        config: DictConfig,
        device_id: int = 0
    ) -> Dict[str, Any]:
        """
        Validate GPU compatibility for configuration settings.
        
        Checks:
        - torch.compile support (requires compute capability >= 7.0)
        - FP16 support (requires compute capability >= 7.0)
        - Batch sizes against GPU memory
        
        Args:
            config: Primary configuration to validate
            device_id: CUDA device ID to check
            
        Returns:
            Dictionary with validation results:
            - warnings: List of warning messages
            - errors: List of error messages
            - fixes_applied: List of automatic fixes applied
        """
        warnings = []
        errors = []
        fixes_applied = []
        
        gpu_checker = _get_gpu_checker(device_id)
        if gpu_checker is None:
            warnings.append("GPU compatibility checker not available, skipping GPU validation")
            return {
                "warnings": warnings,
                "errors": errors,
                "fixes_applied": fixes_applied
            }
        
        if not gpu_checker.is_cuda_available():
            warnings.append("CUDA not available, skipping GPU compatibility checks")
            return {
                "warnings": warnings,
                "errors": errors,
                "fixes_applied": fixes_applied
            }
        
        gpu_info = gpu_checker.get_gpu_info()
        print(f"[INFO] GPU compatibility check: {gpu_info['name']}, "
              f"compute capability {gpu_info['compute_capability']}, "
              f"{gpu_info['total_memory_gb']:.1f}GB")
        
        # Check performance settings
        if 'performance' in config:
            performance = config.performance
            
            # Check torch.compile
            if performance.get('enable_torch_compile', False):
                if not gpu_checker.check_torch_compile_support():
                    error_msg = (
                        f"torch.compile requires compute capability >= 7.0, "
                        f"but GPU has {gpu_info['compute_capability']}"
                    )
                    if config.get('compatibility', {}).get('auto_disable_torch_compile_on_old_gpu', True):
                        performance.enable_torch_compile = False
                        fixes_applied.append(f"Auto-disabled torch.compile: {error_msg}")
                        print(f"[WARN] {fixes_applied[-1]}")
                    else:
                        errors.append(error_msg)
            
            # Check FP16
            if performance.get('use_half_precision', False):
                if not gpu_checker.check_fp16_support():
                    error_msg = (
                        f"FP16 requires compute capability >= 7.0, "
                        f"but GPU has {gpu_info['compute_capability']}"
                    )
                    if config.get('compatibility', {}).get('auto_disable_fp16_on_old_gpu', True):
                        performance.use_half_precision = False
                        fixes_applied.append(f"Auto-disabled FP16: {error_msg}")
                        print(f"[WARN] {fixes_applied[-1]}")
                    else:
                        errors.append(error_msg)
        
        # Validate batch sizes with automatic correction if enabled
        auto_adjust_enabled = config.get('compatibility', {}).get('auto_adjust_batch_size_on_oom', True)
        
        if 'inference' in config:
            inference = config.inference
            if 'batch_size' in inference:
                validation = gpu_checker.validate_batch_size_for_gpu(
                    inference.batch_size,
                    batch_type="inference"
                )
                if validation['warnings']:
                    warnings.extend(validation['warnings'])
                if validation['errors']:
                    errors.extend(validation['errors'])
                    # Auto-fix if enabled and batch_size exceeds safe limit
                    if auto_adjust_enabled:
                        recommended = validation['recommended']
                        inference.batch_size = recommended
                        fixes_applied.append(
                            f"Auto-adjusted inference.batch_size to {recommended} "
                            f"for {validation.get('gpu_memory_gb', 0):.1f}GB GPU"
                        )
                        print(f"[WARN] {fixes_applied[-1]}")
        
        if 'motion_processor' in config:
            motion = config.motion_processor
            if 'batch_size' in motion:
                validation = gpu_checker.validate_batch_size_for_gpu(
                    motion.batch_size,
                    batch_type="inference"
                )
                if validation['warnings']:
                    warnings.extend(validation['warnings'])
                if validation['errors']:
                    errors.extend(validation['errors'])
                    # Auto-fix if enabled and batch_size exceeds safe limit
                    if auto_adjust_enabled:
                        recommended = validation['recommended']
                        motion.batch_size = recommended
                        fixes_applied.append(
                            f"Auto-adjusted motion_processor.batch_size to {recommended} "
                            f"for {validation.get('gpu_memory_gb', 0):.1f}GB GPU"
                        )
                        print(f"[WARN] {fixes_applied[-1]}")
            
            if 'warp_decode_batch_size' in motion:
                validation = gpu_checker.validate_batch_size_for_gpu(
                    motion.warp_decode_batch_size,
                    batch_type="warp_decode"
                )
                if validation['warnings']:
                    warnings.extend(validation['warnings'])
                if validation['errors']:
                    errors.extend(validation['errors'])
                    # Auto-fix if enabled and batch_size exceeds safe limit
                    if auto_adjust_enabled:
                        recommended = validation['recommended']
                        motion.warp_decode_batch_size = recommended
                        fixes_applied.append(
                            f"Auto-adjusted warp_decode_batch_size to {recommended} "
                            f"for {validation.get('gpu_memory_gb', 0):.1f}GB GPU"
                        )
                        print(f"[WARN] {fixes_applied[-1]}")
        
        return {
            "warnings": warnings,
            "errors": errors,
            "fixes_applied": fixes_applied,
            "gpu_info": gpu_info
        }
    
    def apply_gpu_compatibility_fixes(
        self,
        config: DictConfig,
        device_id: int = 0
    ) -> DictConfig:
        """
        Apply automatic GPU compatibility fixes to configuration.
        
        This method modifies the config in place based on GPU capabilities.
        
        Args:
            config: Primary configuration to fix
            device_id: CUDA device ID to check
            
        Returns:
            Modified configuration (same object, modified in place)
        """
        gpu_checker = _get_gpu_checker(device_id)
        if gpu_checker is None or not gpu_checker.is_cuda_available():
            return config
        
        # Validate and get fixes
        validation_result = self.validate_gpu_compatibility(config, device_id)
        
        # Log fixes applied
        if validation_result['fixes_applied']:
            print(f"[INFO] Applied {len(validation_result['fixes_applied'])} GPU compatibility fixes")
        
        # Log warnings
        if validation_result['warnings']:
            print(f"[WARN] {len(validation_result['warnings'])} GPU compatibility warnings:")
            for warn in validation_result['warnings']:
                print(f"  - {warn}")
        
        # Log errors (non-fatal, but should be addressed)
        if validation_result['errors']:
            print(f"[ERROR] {len(validation_result['errors'])} GPU compatibility errors:")
            for err in validation_result['errors']:
                print(f"  - {err}")
        
        return config


# Singleton instance for easy access
_config_manager: Optional[ConfigManager] = None


def get_config_manager(primary_config_path: Optional[str] = None) -> ConfigManager:
    """
    Get or create ConfigManager singleton instance.
    
    Args:
        primary_config_path: Path to primary config file (only used on first call)
        
    Returns:
        ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(primary_config_path)
    return _config_manager

