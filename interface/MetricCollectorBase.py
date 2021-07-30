from typing import List, Optional, Type, Dict, Any, Union
from interface.ModelBase import ModelBase
from abc import ABC, abstractmethod
from numpy import ndarray

class MetricCollectorBase(ABC):

    @classmethod
    def version(cls) -> str: return "1.0"

    @classmethod
    @abstractmethod
    def supported_metrics(cls) -> List[str]:
        """
        Get a list of supported metrics

        Returns
        -------
            metric_names : list with elements of type string
                The names of the visual quality assessment metrics supported by 
                the metrics collector
        """
        ...

    @abstractmethod
    def __init__(
        self, 
        model: Type[ModelBase], 
        metric_names: Optional[List[str]] = None
    ) -> None: 
        """
        Initialize visual quality assessment metrics collector

        Parameters
        ----------
            model : ModelBase
                The model instance to be used in the attack
            metric_names : list with elements of type string
                The names of the metrics to be collected from the internal list that
                maps metric names to metric classes. Default is None
        Raises
        ------
            NameError if an invalid metric name is provided
        Note
        ----
            This method must create the following instance variable:
                self.model = model
        """
        ...

    @abstractmethod
    def __call__(
        self, 
        image: ndarray, 
        adversarial_image: ndarray
    ) -> Dict[str, Union[float, int]]: 
        """
        Collect metrics on samples

        Parameters
        ----------
            image : numpy.ndarray
                Reference image
            adversarial_image : numpy.ndarray
                Adversarial image at current step
        Returns
        -------
            result : dict
                Dictionary with metric names as keys and their evaluation on the input data
                as values
        Note
        ----
            This method must also add the current model query count as to the result dictionary, 
            e.g.: 
                'model queries': self.model.get_query_count()
        """
        ...

    @abstractmethod
    def get_metric_list(self) -> List[str]:
        """
        Get the list of metrics to be collected

        Returns
        -------
            metric_names : list with elements of type string
                The metric names of the metrics to be collected
        """
        ...
