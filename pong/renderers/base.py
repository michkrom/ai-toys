from abc import ABC, abstractmethod

class BaseRenderer(ABC):
    """Base renderer class."""

    def render(self, state, left_controller=None, right_controller=None, l_move=None, r_move=None):
        raise NotImplementedError("Subclasses must implement render()")