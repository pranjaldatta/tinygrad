from .core import Tensor


def MSELoss(y_pred, y, reduction='mean'):
    """
    Mean Squared Error Loss Function.

    Parameters:
        - y_pred: The models prediction
        - y: The truth value

    Returns:
        - A scaler Tensor loss value

    NOTE: y_pred and y should have the same dimensions

    """
    try:
        if len(y_pred) == len(y):
            pass
        else:
            raise Exception
    except Exception:
        print("ERROR: y and y_pred of not same length")
        exit()
    
    losses = [(yi - pi)**2 for yi, pi in zip(y, y_pred)]
    
    if reduction == "mean":
        avg_loss = sum(losses) / (1.0 * len(losses))
    else:
        # not implemented yet. remove when implemented.
        raise NotImplementedError("reduction methods MEAN implemented. Got {}".format(reduction))

    return avg_loss


def MaxMarginLoss(y_pred, y, reduction="mean"):
    """
    SVM Max-Margin Loss.

    Parameters:
        - y_pred: the model's prediction
        - y: the truth value

    Returns:
        - A scaler Tensor loss value

    NOTE: y_pred and y should have the same length
    """
    try:
        if len(y_pred) == len(y):
            pass
        else:
            raise Exception
    except Exception:
        print("ERROR: y and y_pred of not same length")
        exit()

    losses = [(1 + (-yi)*pi).relu() for yi, pi in zip(y, y_pred)]
    
    if reduction == "mean":
        avg_loss = sum(losses) * (1.0/len(losses))
    else:
        raise NotImplementedError("reduction methods MEAN implemented. Got {}".format(reduction))
    
    return avg_loss


def L1Loss(y_pred, y):
    """
    Implements L1 Loss.
 
    Parameters:
        - y_pred: the model's predictions
        - y: the truth values
    
    Returns:
        - A scaler tensor loss value
    """
    try:
        if len(y_pred) == len(y):
            pass
        else:
            raise Exception
    except Exception:
        print("ERROR: y and y_pred of not same length")
        exit()
    
    losses = [(pi-yi).abs() for yi, pi in zip(y, y_pred)]
        
    if reduction == "mean":
        avg_loss = sum(losses) * (1.0/len(losses))
    else:
        raise NotImplementedError("reduction methods MEAN implemented. Got {}".format(reduction))

    return avg_loss


def SmoothL1Loss(y_pred, y, reduction="mean"):
    """
    Implements Smooth L1 Loss. Note: Here we implement the PyTorch version of the SmoothL1Loss.
    i.e. in this version beta = 1. Please check online Pytorch documentation for 
    more clarification

    Parameters:
        - y_pred: the model's prediction
        - y: thee truth values
    
    Returns:
        - A scaler tensor loss value
    """
    try:
        if len(y_pred) == len(y):
            pass
        else:
            raise Exception
    except Exception:
        print("EXCEPTION: y and y_pred of not same length")
 
    # this implementation runs into runtime overflows
    #losses = [0.5*(yi-pi)**2 if (yi-pi).abs().data < 1 else ((yi-pi)-0.5) for pi, yi in zip(y_pred, y)]

    losses = []
    for pi, yi in zip(y_pred, y):

        diff = yi - pi
        diff_abs = diff.abs()
        if diff_abs.data < 1:
            losses += [0.5*diff_abs**2]
        else:
            losses += [diff_abs - 0.5]

    if reduction == "mean":
        avg_loss = sum(losses) * (1.0 / len(losses))
    elif reduction == "sum":
        avg_loss = sum(losses)
    else:
        raise NotImplementedError("reduction methods only MEAN and SUM implemented. Got {}".format(reduction))

    return avg_loss