"""
Base model class for all data models.
"""
from datetime import datetime
from typing import Any, Dict, Optional, Type, TypeVar
from pydantic import BaseModel as PydanticBaseModel, Field

T = TypeVar('T', bound='BaseModel')

class BaseModel(PydanticBaseModel):
    """Base model with common functionality for all models."""
    
    id: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
        arbitrary_types_allowed = True
        validate_assignment = True
    
    def to_dict(self, **kwargs) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return self.dict(**kwargs)
    
    def to_json(self, **kwargs) -> str:
        """Convert model to JSON string."""
        return self.json(**kwargs)
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create model instance from dictionary."""
        return cls.parse_obj(data)
    
    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """Create model instance from JSON string."""
        return cls.parse_raw(json_str)
    
    def update(self, **kwargs) -> None:
        """Update model attributes."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
