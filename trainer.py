#!/usr/bin/env python3

from __future__ import annotations
import sys
import time
import pickle
import copy
import typing

import tqdm
import torch


from utils import DEVICE, Statistics, free_memory, copy_to, GatedStdout

__all__ = ('Trainer',)

# Type alias
TensorOrSequence = (
    torch.Tensor | tuple[torch.Tensor] | list[torch.Tensor]
)

_VALID_SECOND_TIME_WARNING = True


class Trainer():
    """Class for training a model."""

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: torch.nn.Module,
                 *,
                 device: torch.device | int | str | None = None,
                 start_epoch: int = 0,
                 filename: str | None = None,
                 scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
                 forced_gc: bool = False,
                 suppress_display: bool = False
                 ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion          # loss function
        # Use property setter to move model and loss function to target device
        self.device = DEVICE if device is None else torch.device(device)

        self.epoch = start_epoch
        self.filename = 'trainer.trainer' if filename is None else filename
        self.scheduler = scheduler
        self.is_forced_gc = forced_gc
        self.stdout = GatedStdout(suppress_display)

        self.history: dict[str, list[float]] = {'train_loss': [],
                                                'validate_loss': [],
                                                }

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, device: torch.device | int | str) -> Trainer:
        device = torch.device(device)
        torch_vars = copy_to(
            [self.model,
             self.criterion,
             self.optimizer,
             ],
            device)
        (self.model,
         self.criterion,
         self.optimizer,
         ) = torch_vars
        self._device = device
        return self

    @property
    def lr(self) -> list[float]:
        return [pg['lr'] for pg in self.optimizer.param_groups]

    @lr.setter
    def lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def save(self, device: torch.device | int | str = "cpu"):
        """Save trainer object.

        The trainer is saved to the given `device`, along with the
        model. The `device` specified here does not change the device
        type of the trainer instance, but only saves all the variables
        to this `device`. The default target device is "cpu".

        The `device` specified in this method has nothing to do with
        the `load` method's `device` argument. The `device` argument
        is introduced here to solve the problem that a cuda model can
        not be saved and then loaded on another computer without
        gpu. So it is always suggested to set the `device` argument
        here to "cpu" to make sure it can be loaded on any computer.
        """
        device = torch.device(device)
        if device == self.device:
            trainer = self
        else:
            trainer = copy.deepcopy(self)
            trainer.device = device
        with open(self.filename, 'wb') as f:
            f.write(pickle.dumps((trainer.__dict__, self.device)))

    def save_as(self, filename: str):
        self.filename = filename
        return self.save()

    @classmethod
    def load(cls,
             filename: str,
             device: torch.device | int | str | None = None
             ) -> typing.Self:
        """Load a trainer object.

        Load the trainer to given `device`. Specifying `device`
        argument here would also change the loaded trainer's `device`
        property. If device is not given, it is defaulted to the
        object's `device` property.
        """
        with open(filename, 'rb') as f:
            data, default_device = pickle.loads(f.read())
        res = object.__new__(cls)
        res.__dict__.update(data)
        if device and (res.device != torch.device(device)):
            res.device = device
        elif default_device != res.device:
            res.device = default_device
        return res

    def _forward(self, x: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        """Forward pass supporting single tensor or list of tensors.

        Parameters
        ----------
        x : torch.Tensor or list of tensors
            Input to the model. Can be a single tensor or a list
            of tensors that will be unpacked as model arguments.

        Returns
        -------
        torch.Tensor
            Model output.
        """
        if isinstance(x, torch.Tensor):
            return self.model(x)
        return self.model(*x)

    def _move_to_device(
        self,
        data: TensorOrSequence
    ) -> TensorOrSequence:
        """Move data (Tensor, tuple, or list) to the target device."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, tuple):
            return tuple(d.to(self.device) for d in data)
        elif isinstance(data, list):
            return [d.to(self.device) for d in data]
        return data

    def _get_batch_size(
        self,
        data: TensorOrSequence
    ) -> int:
        """Extract batch size from Tensor, tuple, or list."""
        if isinstance(data, torch.Tensor):
            return data.shape[0]
        elif isinstance(data, (tuple, list)):
            # The first element should be a Tensor
            return data[0].shape[0]
        raise TypeError(f"Cannot determine batch size from type {type(data)}")

    def train(
        self,
        loader: torch.utils.data.DataLoader,
        preprocess: typing.Optional[
            typing.Callable[
                [typing.Any, typing.Any],
                tuple[TensorOrSequence, TensorOrSequence]
            ]
        ] = None
    ) -> Statistics[float]:
        """Train the model by given dataloader.

        Args:
            loader: The dataloader providing training data.
            preprocess: Optional callable to preprocess input data
                `x` and target `y`. It takes `x` and `y` as inputs
                and returns the processed `x` and `y`. Both `x` and
                `y` can be a Tensor, a tuple of Tensors, or a list
                of Tensors. This is essential for tasks like Flow
                Matching or Diffusion where inputs (e.g., noisy
                images, timesteps) and targets (e.g., velocity
                fields) are constructed dynamically from the raw
                data.

        """
        t_start = time.time()
        self.model.train()
        tq = tqdm.tqdm(loader,
                       desc="train",
                       ncols=None,
                       leave=False,
                       file=self.stdout,
                       unit="batch")
        loss_meter: Statistics[float] = Statistics()
        # User defined preprocess
        additional_data = self.additional_train_preprocess(tq)
        for x, y in tq:
            if preprocess:
                x, y = preprocess(x, y)

            current_batch_size = self._get_batch_size(y)

            # Move data to device
            x = self._move_to_device(x)
            y = self._move_to_device(y)

            # compute prediction error
            y_pred = self._forward(x)
            loss = self.criterion(y_pred, y)
            # backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # record results
            loss_meter.update(loss.item(), current_batch_size)
            tq.set_postfix(loss=f"{loss_meter.value:.4g}")

            # Do some user-defined process.
            self.additional_train_process(additional_data,
                                          y_pred, y, loss, tq)

            # Free some space before next round.
            del x, y, y_pred, loss
            if self.is_forced_gc:
                free_memory()

        # Save information for this epoch.
        self.epoch += 1
        self.history['train_loss'].append(loss_meter.average)

        # User_defined postprocess.
        self.additional_train_postprocess(additional_data)

        print(f'train result [{self.epoch}]: '
              f'avg loss = {loss_meter.average:.4g}, '
              f'wall time = {time.time() - t_start:.2f}s',
              file=self.stdout)
        
        if self.scheduler:
            self.scheduler.step()

        return loss_meter

    def validate(
        self,
        loader: torch.utils.data.DataLoader,
        preprocess: typing.Optional[
            typing.Callable[
                [typing.Any, typing.Any],
                tuple[TensorOrSequence, TensorOrSequence]
            ]
        ] = None
    ) -> Statistics[float]:
        """Validate the model.

        Args:
            loader: The dataloader providing validation data.
            preprocess: Optional callable to preprocess input data
                `x` and target `y`. It takes `x` and `y` as inputs
                and returns the processed `x` and `y`. Both `x` and
                `y` can be a Tensor, a tuple of Tensors, or a list
                of Tensors. Useful for calculating validation loss
                in diffusion models where targets depend on sampled
                timesteps.
        """
        t_start = time.time()
        self.model.eval()
        tq = tqdm.tqdm(loader,
                       desc="valid",
                       ncols=None,
                       leave=False,
                       file=self.stdout,
                       unit="batch")
        loss_meter: Statistics[float] = Statistics()
        # User-defined preprocess
        additional_data = self.additional_validate_preprocess(tq)

        for x, y in tq:
            if preprocess:
                x, y = preprocess(x, y)

            current_batch_size = self._get_batch_size(y)

            # Move data to device
            x = self._move_to_device(x)
            y = self._move_to_device(y)

            with torch.no_grad():
                y_pred = self._forward(x)
            loss = self.criterion(y_pred, y)
            loss_meter.update(loss.item(), current_batch_size)
            tq.set_postfix(loss=f"{loss_meter.value:.4g}")

            # Do some user-defined process.
            self.additional_validate_process(additional_data,
                                             y_pred, y, loss, tq)

            del x, y, y_pred, loss
            if self.is_forced_gc:
                free_memory()

        # Save validation results only the fisrt run.
        if len(self.history['validate_loss']) < self.epoch:
            self.history['validate_loss'].append(loss_meter.average)
        else:
            global _VALID_SECOND_TIME_WARNING
            if _VALID_SECOND_TIME_WARNING:
                sys.stderr.write("The model is validated for the "
                                 "second time in the same epoch, "
                                 "validation result will not be "
                                 "recorded. "
                                 "This warning will be "
                                 "turned off in this session.\n")
                _VALID_SECOND_TIME_WARNING = False

        # User_defined postprocess.
        self.additional_validate_postprocess(additional_data)

        print(f'valid result [{self.epoch}]: '
              f'avg loss = {loss_meter.average:.4g}, '
              f'wall time = {time.time() - t_start:.2f}s',
              file=self.stdout)
        return loss_meter

    def step(self,
             train_dataloader: torch.utils.data.DataLoader,
             valid_dataloader: torch.utils.data.DataLoader,
             save_trainer: bool = True,
             save_best_model: str | None = None,
             ) -> bool:
        """Train and validate the model, return if it is the best model.

        Note that if `save_best_model` is enabled, it selects the best
        model by its validation loss, which might not be good for
        classification problems.

        Parameters
        ----------
        save_trainer: bool, optional
            whether to save the trainer after this epoch. defaults to
            True.
        save_best_model: str, optional
            Filename for saving the model if it is the best model
            indicated by the validation procedure. If not given, the
            model will not be saved automatically.

            """
        print(f'    ---- Epoch {self.epoch} ----    ', file=self.stdout)
        self.train(train_dataloader)
        loss = self.validate(valid_dataloader)
        if save_trainer:
            self.save()
        if loss == min(self.history['validate_loss']):
            if save_best_model:
                print('This model will be saved as the best model.',
                      file=self.stdout)
                with open(save_best_model, 'wb') as f:
                    torch.save(self.model, f)
            return True
        return False

    # Class methods can be overwritten.

    def additional_train_preprocess(self, tq: tqdm.std.tqdm) -> typing.Any:
        """Additional pre-process in each epoch.

        This method can be overwritten to do some additional work
        before iterations in each epoch (eg, prepare dict for
        `additional_train_process`). The return value of the method
        will be used for `additional_train_process` and
        `additional_train_postprocess`

        Parameters
        ----------
        tq: tqdm object
            The tqdm object. Can modify display here.    

        Return
        ------
        additional_data: typing.Any
            The returned data will be used processed in each batch
            and the end of the epoch.
        """
        return None

    def additional_train_process(self,
                                 additional_data: typing.Any,
                                 y_pred: torch.Tensor,
                                 y_true: torch.Tensor,
                                 loss: torch.Tensor,
                                 tq: tqdm.std.tqdm):
        """Additional process after training the model in each batch.

        This method can be overwritten to do some additional work
        after the loss is calculated in each batch (eg, calculate
        top-k error in classification task). This function has no
        return value.

        Parameters
        ----------
        additional_data: typing.Any
            Defined in `additional_train_preprocess`.
        y_pred: torch.Tensor
            Predicted value by model.
        y_true: torch.Tensor
            True value given by train dataset.
        loss: torch.Tensor
            Loss of this batch given by criterion.
        tq: tqdm object
            Can modify display here.
        """
        pass

    def additional_train_postprocess(self,
                                     additional_data: typing.Any):
        """Additional postprocess in each epoch.

        This method can be overwritten to do some additional work
        after the epoch is finished (eg. saving `additional_data`).

        Parameters
        ----------
        additional_data: typing.Any
            Defined in `additional_train_preprocess`.
        """
        pass

    def additional_validate_preprocess(self, tq: tqdm.std.tqdm) -> typing.Any:
        """Additional pre-process in each epoch.

        This method can be overwritten to do some additional work
        before iterations in each epoch (eg, prepare dict for
        `additional_validate_process`). The return value of the method
        will be used for `additional_validate_process` and
        `additional_validate_postprocess`

        Parameters
        ----------
        tq: tqdm object
            The tqdm object. Can modify display here.

        Return
        ------
        additional_data: typing.Any
            The returned data will be used processed in each batch
            and the end of the epoch.
        """
        return None

    def additional_validate_process(self,
                                    additional_data: typing.Any,
                                    y_pred: torch.Tensor,
                                    y_true: torch.Tensor,
                                    loss: torch.Tensor,
                                    tq: tqdm.std.tqdm):
        """Additional process after model validation in each batch.

        This method can be overwritten to do some additional work
        after the loss is calculated in each batch (eg, calculate
        top-k error in classification task). This function has no
        return value.

        Parameters
        ----------
        additional_data: typing.Any
            Defined in `additional_validate_preprocess`.
        y_pred: torch.Tensor
            Predicted value by model.
        y_true: torch.Tensor
            True value given by validation dataset.
        loss: torch.Tensor
            Loss of this batch given by criterion.
        tq: tqdm object
            Can modify display here.
        """
        pass

    def additional_validate_postprocess(self,
                                        additional_data: typing.Any):
        """Additional postprocess in each epoch.

        This method can be overwritten to do some additional work
        after the epoch is finished (eg. saving `additional_data`).

        Parameters
        ----------
        additional_data: typing.Any
            Defined in `additional_validate_preprocess`.
        """
        pass
