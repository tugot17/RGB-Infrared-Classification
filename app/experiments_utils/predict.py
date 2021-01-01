import torch


@torch.no_grad()
def predict(model, dataloader, device, get_x_method, num_classes):
    """
    :param model:
    :param dataloader:
    :param device:
    :param get_x_method:
    :return:
    """
    model = model.to(device)

    num_elements = len(dataloader.dataset)
    num_batches = len(dataloader)

    batch_size = dataloader.batch_size

    predictions = torch.zeros(num_elements, num_classes)

    for i, batch in enumerate(dataloader):
        start = i * batch_size
        end = start + batch_size
        if i == num_batches - 1:
            end = num_elements

        x = get_x_method(batch)

        # case for rgb infrared as two seprete inputs
        if isinstance(x, tuple):
            x = (element.to(device) for element in x)
        else:
            x = x.to(device)

        logits = model(x)
        predictions[start:end] = logits.detach().cpu()

    return predictions
