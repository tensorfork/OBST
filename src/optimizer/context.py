import typing

import mesh_tensorflow as mtf

from ..dataclass import ModelParameter


class OptimizerCtx:
    def __init__(self, op: mtf.Operation, grad_outputs: typing.List[mtf.Tensor], downstream: typing.Set[mtf.Operation],
                 tensor_to_gradient: dict, tensor_to_var: dict, params: ModelParameter, loss_idx: int, update_ops: list,
                 debug_gradients_dict: dict, loss_list: list, first_grad: dict,
                 loss_1__loss_1: typing.Optional[mtf.Tensor], loss_1__loss_2: typing.Optional[mtf.Tensor],
                 loss_2__loss_2: typing.Optional[mtf.Tensor], mstep: mtf.Tensor, step: mtf.Tensor, neg_step,
                 dtype: mtf.VariableDType, beta1: mtf.Tensor, beta2: mtf.Tensor, learning_rate: mtf.Tensor,
                 step_count: mtf.Tensor):
        self.step_count = step_count
        self.op = op
        self.grad_outputs = grad_outputs
        self.tensor_to_gradient = tensor_to_gradient
        self.tensor_to_var = tensor_to_var
        self.params = params
        self.loss_idx = loss_idx
        self.update_ops = update_ops
        self.debug_gradients_dict = debug_gradients_dict
        self.loss_list = loss_list
        self.first_grad = first_grad
        self.loss_1__loss_1 = loss_1__loss_1
        self.loss_1__loss_2 = loss_1__loss_2
        self.loss_2__loss_2 = loss_2__loss_2
        self.mstep = mstep
        self.step = step
        self.dtype = dtype
        self.neg_step = neg_step
        self.beta1 = beta1
        self.beta2 = beta2
        self.learning_rate = learning_rate
        self.args = [op, grad_outputs, downstream, tensor_to_gradient, tensor_to_var, params, loss_idx, update_ops,
                     debug_gradients_dict, loss_list, first_grad, loss_1__loss_1, loss_1__loss_2, loss_2__loss_2, mstep,
                     step, dtype, beta1, beta2, learning_rate]

        self.var: typing.Optional[mtf.Variable] = None
        self.tensor: typing.Optional[mtf.Tensor] = None
        self.grad_buffer: typing.Optional[mtf.Variable] = None
        self.grad: typing.Optional[mtf.Tensor] = None
        self.original_grad: typing.Optional[mtf.Tensor] = None
        self.variable_to_gradient: typing.Optional[typing.Dict[mtf.Variable:mtf.Tensor]] = {}

        self.global_norm_reciprocal: typing.Optional[mtf.Tensor] = None

    def __call__(self, tensor: mtf.Tensor, var: mtf.Variable, grad: mtf.Tensor):
        self.var = var
        self.tensor = tensor
        self.grad = self.original_grad = grad
        self.op = self.tensor_to_gradient[tensor][3]
        return self
