import torch.autograd

import utils

log = utils.log.get_logger()


def gradient_penalty(inputs, output, weight=10):
    if weight == 0:
        return torch.zeros_like(output).sum()
    elif not inputs.requires_grad:
        log.debug("inputs.requires_grad is False, skip gradient penalty")
        return torch.zeros_like(output).sum()

    gradients = torch.autograd.grad(
        outputs=output,
        inputs=inputs,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.reshape(len(gradients), -1)
    return weight * ((torch.linalg.vector_norm(gradients, dim=1) - 1) ** 2).mean()
