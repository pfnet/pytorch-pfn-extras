import torch


class AccuracyMetric:
    """A metric for an evaluator to report accuracy.

    Args:
        label_key (str): The key name of label.
        output_key (str): The key name of prediction.

    .. seealso:
       :func:`pytorch_pfn_extras.engine.create_evaluator`
    """
    def __init__(self, label_key, output_key):
        self.label_key = label_key
        self.output_key = output_key

    def _preprocess_input(self, batch, out):
        labels = batch[self.label_key].cpu()
        n_output = labels.shape[0]
        pred = out[self.output_key][:n_output].cpu()
        return labels, n_output, pred

    def __call__(self, batch, out):
        with torch.no_grad():
            labels, n_output, pred = self._preprocess_input(batch, out)
            correct = (labels.view_as(pred) == pred).sum().item()
        return {"accuracy": correct / n_output}
