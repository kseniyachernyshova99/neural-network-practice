"""Application-specific validation errors."""
class PracticeError(Exception):
    """Base error displayed by the console interface."""


class InvalidLayerSizeError(PracticeError):
    """Invalid layer configuration."""


class MismatchedDataError(PracticeError):
    """Data do not match the expected format."""


class InputError(PracticeError):
    """Invalid user input or file operation."""
