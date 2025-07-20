"""
Advanced state machine for order lifecycle management.

Implements a hierarchical state machine with validation,
audit logging, and transition hooks.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, validator

from core.utils.distributed_lock import DistributedLock, order_lock

T = TypeVar('T', bound='StateMachine')

class StateTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""
    def __init__(self, from_state: str, to_state: str, reason: str = ""):
        self.from_state = from_state
        self.to_state = to_state
        self.reason = reason
        super().__init__(
            f"Invalid transition from {from_state} to {to_state}. {reason}"
        )

class StateValidationError(Exception):
    """Raised when a state fails validation."""
    pass

@dataclass
class StateTransition:
    """Represents a state transition with metadata."""
    from_state: str
    to_state: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    actor: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    transition_id: str = field(default_factory=lambda: str(uuid4()))

class StateMachineMeta(type):
    """Metaclass for state machines that tracks valid transitions."""
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Initialize transition maps if not present
        if not hasattr(cls, '_transitions'):
            cls._transitions = {}
            
        # Collect transitions from class attributes
        transitions = {}
        for key, value in namespace.items():
            if key.startswith('TRANSITION_'):
                from_state, to_state = value
                transitions[(from_state, to_state)] = key
                
        # Update transitions with any new ones
        cls._transitions.update(transitions)
        
        return cls

class StateMachine(metaclass=StateMachineMeta):
    """Base class for state machines with validation and auditing."""
    
    # Define valid state transitions as class variables
    # Format: TRANSITION_<NAME> = (from_state, to_state)
    
    def __init__(self, initial_state: str, state_field: str = 'state'):
        """Initialize the state machine."""
        self._state_field = state_field
        self._state = initial_state
        self._transition_history: List[StateTransition] = []
        self._transition_lock = asyncio.Lock()
        self._redis = None  # Will be set by set_redis
        
    @property
    def state(self) -> str:
        """Get the current state."""
        return self._state
    
    @property
    def transition_history(self) -> List[StateTransition]:
        """Get the transition history."""
        return self._transition_history.copy()
    
    def set_redis(self, redis):
        """Set the Redis client for distributed locking."""
        self._redis = redis
    
    async def can_transition(self, to_state: str) -> bool:
        """Check if a transition to the target state is valid."""
        return (self._state, to_state) in self._transitions
    
    async def validate_transition(self, to_state: str, **kwargs) -> None:
        """Validate a state transition.
        
        Raises:
            StateTransitionError: If the transition is invalid
        """
        if (self._state, to_state) not in self._transitions:
            raise StateTransitionError(
                self._state,
                to_state,
                "No such transition defined"
            )
            
        # Call any validation hooks
        validate_method = f"validate_{self._state.lower()}_to_{to_state.lower()}"
        if hasattr(self, validate_method):
            validator = getattr(self, validate_method)
            await validator(**kwargs)
    
    async def on_transition(
        self,
        from_state: str,
        to_state: str,
        **kwargs
    ) -> None:
        """Called after a successful state transition.
        
        Subclasses can override this to add custom behavior.
        """
        pass
    
    async def transition_to(
        self,
        to_state: str,
        actor: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Transition to a new state with validation and locking.
        
        Args:
            to_state: The target state
            actor: Optional identifier for who initiated the transition
            **kwargs: Additional data for the transition
            
        Returns:
            bool: True if the transition was successful, False otherwise
            
        Raises:
            StateTransitionError: If the transition is invalid
        """
        # Use distributed lock if Redis is available
        if self._redis:
            async with order_lock(self._redis, f"statemachine:{id(self)}"):
                return await self._do_transition(to_state, actor, **kwargs)
        else:
            async with self._transition_lock:
                return await self._do_transition(to_state, actor, **kwargs)
    
    async def _do_transition(
        self,
        to_state: str,
        actor: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Internal method to perform the state transition."""
        from_state = self._state
        
        # Validate the transition
        await self.validate_transition(to_state, **kwargs)
        
        # Create transition record
        transition = StateTransition(
            from_state=from_state,
            to_state=to_state,
            actor=actor,
            metadata=kwargs
        )
        
        # Update state
        self._state = to_state
        
        # Record the transition
        self._transition_history.append(transition)
        
        # Call transition hook
        await self.on_transition(from_state, to_state, **kwargs)
        
        logger.info(
            f"State transition: {from_state} → {to_state} "
            f"(actor={actor or 'system'})"
        )
        
        return True
    
    def get_valid_transitions(self) -> Set[str]:
        """Get all valid target states from the current state."""
        return {
            to_state for (from_state, to_state) in self._transitions
            if from_state == self._state
        }
