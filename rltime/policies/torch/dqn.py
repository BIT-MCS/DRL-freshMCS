import torch
import torch.nn as nn
import numpy as np

from .torch_policy import TorchPolicy
from rltime.models.torch.utils import linear
import logging
import gym


class DQNPolicy(TorchPolicy):
    """The basic DQN policy, with dueling-layer support.

    This is also the base-class for distributional policies like DistDQN and
    IQN
    """

    def __init__(self, action_space, dueling=False,
                 dueling_value_layer_hidden_size=None, **kwargs):
        """Initializes the policy

        Args:
            action_space: The action space of the ENVs, only 'Discrete' is
                supported for DQN policies
            dueling: Whether to use a dueling architecture (Parallel advantage
                and value layers). Dueling does not add an additional hidden
                layer to the model, rather the last layer of the model will be
                treated as the hidden 'advantage layer', while an additional
                'state-value' layer will be added here in parallel to the last
                layer from the model.
            dueling_value_layer_hidden_size: Size of the hidden value-layer FC
                module. In case of dueling=True the last layer of the model
                will be the 'advantage hidden layer', while this value defines
                the size of the parallel state-value layer. By default (None)
                it will be auto-set to be the same size as the last layer in
                the model (i.e. the 'advantage hidden layer').
                Note that if the last layer of the model is an LSTM and not an
                FC layer, the value-hidden-layer here will still be an FC
                layer, while the 'advantage layer' will be that LSTM layer from
                the model, which may or may not be the intention. If not the
                intention you should add an FC layer in the model after the
                LSTM layer. For example you can use a CNN->LSTM->FC model or
                CNN->FC->LSTM->FC model.
        """
        super().__init__(**kwargs)

        self.branch = 0
        # Some sub-classes need multiple outputs per action (e.g.
        # distributional DQN)
        if isinstance(action_space, gym.spaces.Discrete):
            out_size = action_space.n * self._outputs_per_action()
            # The final output FC layer generating the action-Q values
            self.out_layer = linear(self.model.out_size, out_size)

        elif isinstance(action_space, gym.spaces.Box):
            self.branch = action_space.shape[0] * self._outputs_per_action()
            self.out_layer = nn.ModuleList([linear(self.model.out_size, self.bins) for _ in range(self.branch)])  # branch

        else:
            raise NotImplementedError

        if dueling:
            # For dueling architecture, we inject the value estimation layer
            # before the last layer of the model, in parallel to it
            inner_layer_size = int(np.prod(self.model.get_layer_in_shape(-1)))
            if not dueling_value_layer_hidden_size:
                # Auto-set dueling value-layer hidden size to be same as the
                # parallel action/advanatage layer hidden size
                dueling_value_layer_hidden_size = \
                    int(np.prod(self.model.get_layer_out_shape(-1)))
            logging.getLogger().info(
                "Dueling value hidden layer size: "
                f"{inner_layer_size}x{dueling_value_layer_hidden_size}")

            self.value_hidden_layer = linear(
                inner_layer_size, dueling_value_layer_hidden_size)
            self.value_layer = linear(
                dueling_value_layer_hidden_size, self._outputs_per_action())
        else:
            self.value_layer = None

    def _outputs_per_action(self):
        # By default DQN policy generates 1 output per action
        return 1

    def _process_dueling(self, action_outputs, state_layer):
        """Applies the dueling value-layer and merges with the action-outputs
        (i.e. the 'advantage layer' in this case)"""
        state_layer = state_layer.view(state_layer.shape[0], -1)  # flatten
        # Calculate the state-value estimation using the dedicated value layers
        state_value = self.value_hidden_layer(state_layer)
        state_value = torch.nn.functional.relu(state_value)
        state_value = self.value_layer(
            state_value)  # (batch_size, outputs_per_action)

        state_value = self._shape_value_outputs(state_value)

        if self.branch>0:
            return state_value.unsqueeze(3) + action_outputs - action_outputs.mean(3, keepdim=True)
        else:
            return state_value + action_outputs - action_outputs.mean(2, keepdim=True)  # advantages(dueling)

    def _predict_postprocess(self, output, model_output):
        # Process dueling layer if defined
        if self.value_layer is not None:
            assert ("layer_inputs" in model_output), \
                "Dueling DQN requires the model to output also the inner " \
                "layer inputs"
            # The parallel dueling layer branches out in parallel to the last
            # model layer
            output = self._process_dueling(
                output, model_output['layer_inputs'][-1])
        return output

    def predict(self, x, timesteps):
        # Perform a forward pass on the model
        res = self.model(x, timesteps)
        # Apply the actions linear layer to the output
        if self.branch>0:
            output= torch.stack([l(res['output']) for l in self.out_layer],dim=1)
            output = self._shape_action_outputs(output)
        else:
            output = self.out_layer(res['output'])
            # Reshape the output so that actions are on an independent axis (in
            # case of a distributional policy)
            output = self._shape_action_outputs(output)

        # Post-process the output, this may include dueling layer addition if
        # active, and additional processing by subclasses
        return self._predict_postprocess(output, res)

    def _shape_action_outputs(self, output):
        """Reshapes the outputs, if relevant, to isolate actions to a dedicated
        dimension (For example in case of distributional or IQN)

        Should return the reshaped output and which dim the actions are now on
        """
        # For the basic DQN case the output is already correctly shaped
        # <batch_size, actions>
        return output

    def _shape_value_outputs(self, output):
        return output

    def _actor_predict_postprocess(self, pred):
        """Optional postprocessing of actor predictions in a subclass

        For example for distributional-like subclasses may need to average/sum
        one of the dims or apply some additional processing.
        """
        return pred

    def actor_predict(self, inp, timesteps, for_eval=False):
        # Perform a prediction to get the action q-values
        pred = self.predict(inp, timesteps)

        # Optional postprocessing of the sample prediction by a subclass(For
        # example in case of distributional actions -> reduce them in here)  删掉 Quantiles
        qvalues = self._actor_predict_postprocess(pred)
        qvalues = qvalues.data.cpu().numpy()

        # For DQN, actor action-selection is just argmax on the action qvalues
        # We return the qvalues too though they aren't currently used but
        # might be usefull for helping initialize replay priorities
        if self.branch>0:
            return {
                "actions": np.argmax(qvalues, axis=-1),
                "qvalues": qvalues
            }
        else:
            return {
                "actions": np.argmax(qvalues, axis=-1),
                "qvalues": qvalues
            }
