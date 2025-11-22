import numpy as np

seed = 22102017
rng = np.random.RandomState(seed)


class L1Penalty(object):
    """L1 parameter penalty.

    Term to add to the objective function penalising parameters
    based on their L1 norm.
    """

    def __init__(self, coefficient):
        """Create a new L1 penalty object.

        Args:
            coefficient: Positive constant to scale penalty term by.
        """
        assert coefficient > 0., 'Penalty coefficient must be positive.'
        self.coefficient = coefficient

    def __call__(self, parameter):
        """Calculate L1 penalty value for a parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty term.
        """
        return self.coefficient * np.sum(np.abs(parameter))

    def grad(self, parameter):
        """Calculate the penalty gradient with respect to the parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty gradient with respect to parameter. This
            should be an array of the same shape as the parameter.
        """
        return self.coefficient * np.sign(parameter)

    def __repr__(self):
        return 'L1Penalty({0})'.format(self.coefficient)


class L2Penalty(object):
    """L1 parameter penalty.

    Term to add to the objective function penalising parameters
    based on their L2 norm.
    """

    def __init__(self, coefficient):
        """Create a new L2 penalty object.

        Args:
            coefficient: Positive constant to scale penalty term by.
        """
        assert coefficient > 0., 'Penalty coefficient must be positive.'
        self.coefficient = coefficient

    def __call__(self, parameter):
        """Calculate L2 penalty value for a parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty term.
        """
        return self.coefficient * np.sum(parameter**2)

    def grad(self, parameter):
        """Calculate the penalty gradient with respect to the parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty gradient with respect to parameter. This
            should be an array of the same shape as the parameter.
        """
        return 2 * self.coefficient * parameter

    def __repr__(self):
        return 'L2Penalty({0})'.format(self.coefficient)

class L1L2MixPenalty(object):
    """L1 & L2 mix penalty.
    """

    def __init__(self, l1_coefficient, l2_coefficient):
        """Create a new L1 & L2 mix penalty object.

        Args:
            l1_coefficient: Positive constant for L1 penalty (controls sparsity).
            l2_coefficient: Positive constant for L2 penalty (controls shrinkage).
        """
        assert l1_coefficient >= 0., 'L1 coefficient must be non-negative.'
        assert l2_coefficient >= 0., 'L2 coefficient must be non-negative.'
        assert l1_coefficient > 0. or l2_coefficient > 0., 'At least one coefficient must be positive.'
        
        self.l1_coefficient = l1_coefficient
        self.l2_coefficient = l2_coefficient

    def __call__(self, parameter):
        """Calculate L1 & L2 mix penalty value for a parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty term.
        """
        l1_penalty = self.l1_coefficient * np.sum(np.abs(parameter))
        l2_penalty = self.l2_coefficient * np.sum(parameter ** 2)
        return l1_penalty + l2_penalty

    def grad(self, parameter):
        """Calculate the penalty gradient with respect to the parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty gradient with respect to parameter. This
            should be an array of the same shape as the parameter.
        """
        l1_grad = self.l1_coefficient * np.sign(parameter)
        l2_grad = 2 * self.l2_coefficient * parameter
        return l1_grad + l2_grad

    def __repr__(self):
        return f'L1L2MixPenalty(l1={self.l1_coefficient}, l2={self.l2_coefficient})'
