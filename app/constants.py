"""
Field-length limits shared between SQLAlchemy models (app/models/) and
their matching Pydantic schemas (app/schemas/), so a change only needs
to happen in one place instead of being kept in sync by hand across both.
"""

CUSTOMER_NAME_MAX_LENGTH = 100

PROMPT_NAME_MAX_LENGTH = 200
PROMPT_TEXT_MAX_LENGTH = 5000
