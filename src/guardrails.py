import re
from typing import Optional

class InputSanitizer:
    """Sanitize user input before processing."""

    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous\s+instructions)?",
        r"forget\s+(all\s+)?(previous\s+instructions)?",
        r"new\s+instructions:",
        r"system\s*prompt",
        r"---\s*end\s*(of)?\s*prompt",
        r"pretend\s+you\s+are",
        r"act\s+as\s+(if\s+)?you",
        r"bypass\s+(all\s+)?restrictions",
    ]

    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]

    def is_suspicious(self, text: str) -> tuple[bool, Optional[str]]:
        """Check if input contains suspicious patterns."""
        for pattern in self.patterns:
            if pattern.search(text):
                return True, f"Suspicious pattern detected: {pattern.pattern}"
        return False, None

    def sanitize(self, text: str) -> str:
        """Remove potentially dangerous content."""
        # Remove common injection delimiters
        text = re.sub(r"[-]{3,}", "", text)
        text = re.sub(r"[=]{3,}", "", text)

        # Escape special characters that might confuse the model
        text = text.replace("{{", "{ {").replace("}}", "} }")

        return text.strip()


class SQLValidationError(ValueError):
    """Raised when an SQL query fails safety or structural checks."""
    pass

def safe_sql(query: str) -> str:
    q = query.strip()
    
    # Check for multiple statements (allow at most one trailing semicolon)
    if q.count(';') > 1:
        raise SQLValidationError("Multiple statements are not allowed.")
        
    if not q.lower().startswith('select'):
        raise SQLValidationError("Only SELECT statements are allowed.")
        
    return q