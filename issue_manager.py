from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from issue import BlockingIssue

if TYPE_CHECKING:
    from issue import Issue


class IssueManager:
    """
    A manager class that handles storing, querying, and managing issues.
    Implements Observable pattern to notify UI of issue changes.
    """

    def __init__(self) -> None:
        self._issues: list[Issue] = []
        self._observers: list[Callable] = []

    def register_observer(self, observer: Callable) -> None:
        """Register a function to be called when issues change."""
        if observer not in self._observers:
            self._observers.append(observer)

    def unregister_observer(self, observer: Callable) -> None:
        """Remove an observer."""
        if observer in self._observers:
            self._observers.remove(observer)

    def notify_observers(self) -> None:
        """Notify all observers of a change in issues."""
        for observer in self._observers:
            observer()

    def add_issue(self, issue: Issue) -> None:
        """Add an issue, replacing any existing issues of the same class."""
        self.remove_issues_of_class(issue.__class__)
        self._issues.append(issue)
        self.notify_observers()

    def remove_issue(self, issue: Issue) -> None:
        """Remove a specific issue instance."""
        if issue in self._issues:
            self._issues.remove(issue)
            self.notify_observers()

    def remove_issues_of_class(self, issue_class: type[Issue]) -> None:
        """Remove all issues of a specific class."""
        self._issues = [i for i in self._issues if not isinstance(i, issue_class)]
        self.notify_observers()

    def remove_all_issues(self) -> None:
        """Remove all issues."""
        self._issues.clear()
        self.notify_observers()

    def has_issues(self) -> bool:
        """Check if there are any issues."""
        return len(self._issues) > 0

    def has_blocking_issues(self) -> bool:
        """Check if there are any issues that prevent continuation."""
        return any(isinstance(issue, BlockingIssue) for issue in self._issues)

    def has_issue_of_class(self, issue_class: type[Issue]) -> bool:
        """Check if there is an issue of the specified class."""
        return any(isinstance(issue, issue_class) for issue in self._issues)

    def get_issues(self) -> list[Issue]:
        """Get all issues."""
        return self._issues.copy()

    def get_first_blocking_issue(self) -> Issue | None:
        """Get the first blocking issue, if any."""
        blocking_issues = [issue for issue in self._issues if isinstance(issue, BlockingIssue)]
        return blocking_issues[0] if blocking_issues else None

    def get_most_important_issue(self) -> Issue | None:
        """Get the most important issue (blocking first, then any)."""
        blocking_issue = self.get_first_blocking_issue()
        if blocking_issue:
            return blocking_issue
        return self._issues[0] if self._issues else None
