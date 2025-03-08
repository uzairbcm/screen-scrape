from abc import ABC, abstractmethod


class Issue(ABC):
    """Base abstract class for all issues."""

    def __init__(self, description: str) -> None:
        self.description = description

    def get_styled_message(self) -> tuple[str, str]:
        """Returns a tuple of (message, style)"""
        message = self.get_message()
        return (message, self.get_style())

    @abstractmethod
    def get_message(self) -> str:
        """Returns a user-friendly message describing the issue."""

    @abstractmethod
    def get_style(self) -> str:
        """Returns CSS styling for the issue message."""


class BlockingIssue(Issue):
    """An issue that must be resolved before the user can continue.
    These are critical errors that prevent further progress."""

    def get_style(self) -> str:
        return "background-color:rgb(255,0,0)"


class NonBlockingIssue(Issue):
    """An issue that allows the user to continue without resolving it.
    These are typically warnings that don't prevent further progress."""

    def get_style(self) -> str:
        return "background-color:rgb(255,179,179)"


# Graph Issues
class GraphDetectionIssue(BlockingIssue):
    def get_message(self) -> str:
        return "A graph detection issue occurred. To start reselecting, first click the upper left corner of the graph in the left image."


# Title Issues
class TitleMissingIssue(BlockingIssue):
    def get_message(self) -> str:
        return (
            "A title issue occurred. Please enter the title correctly and click Next/Save when finished.\n"
            "If the title is for the daily view, please either skip enter 'Daily Total'."
        )

    def get_style(self) -> str:
        return "background-color:rgb(255,165,0)"


# Total Issues - Base class
class TotalIssue(Issue):
    """Base class for all total time discrepancy issues."""

    def get_message(self) -> str:
        continuation_message = (
            "You can proceed or reselect the graph for better accuracy."
            if isinstance(self, NonBlockingIssue)
            else "Please reselect the graph to better approximate the total."
        )
        return f"A total time discrepancy issue occurred: {self.description}\n{continuation_message}"


# Total Issues - Specific types
class TotalNotFoundIssue(TotalIssue, NonBlockingIssue):
    """Total usage duration not found in the image."""


class TotalParseErrorIssue(TotalIssue, NonBlockingIssue):
    """Total usage duration unable to be parsed from the image."""


class TotalUnderestimationSmallIssue(TotalIssue, NonBlockingIssue):
    """Underestimation by less than 2 minutes."""


class TotalUnderestimationLargeIssue(TotalIssue, BlockingIssue):
    """Underestimation by 2 or more minutes."""


class TotalOverestimationSmallIssue(TotalIssue, NonBlockingIssue):
    """Overestimation by exactly 1 minute."""


class TotalOverestimationLargeIssue(TotalIssue, BlockingIssue):
    """Overestimation by more than 1 minute."""
