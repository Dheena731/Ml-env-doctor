"""Custom exceptions for ML Environment Doctor."""


class MLEnvDoctorError(Exception):
    """Base exception for ML Environment Doctor."""

    def __init__(self, message: str, suggestion: str = ""):
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion

    def __str__(self) -> str:
        if self.suggestion:
            return f"{self.message}\n💡 Suggestion: {self.suggestion}"
        return self.message


class DiagnosticError(MLEnvDoctorError):
    """Error during diagnostic checks."""

    pass


class FixError(MLEnvDoctorError):
    """Error during auto-fix operations."""

    pass


class DockerError(MLEnvDoctorError):
    """Error during Docker operations."""

    pass


class GPUError(MLEnvDoctorError):
    """Error related to GPU operations."""

    pass


class ConfigurationError(MLEnvDoctorError):
    """Error in configuration."""

    pass


class InstallationError(MLEnvDoctorError):
    """Error during package installation."""

    pass
