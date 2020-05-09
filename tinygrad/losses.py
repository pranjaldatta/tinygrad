from .core import Tensor

def MSELoss(y_pred, y):
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
    
    avg_loss = sum(losses) / (1.0 * len(losses))

    return avg_loss

def MaxMarginLoss(y_pred, y):
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
    
    avg_loss = sum(losses) * (1.0/len(losses))
    
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
        
    avg_loss = sum(losses) * (1.0 / len(losses))

    return avg_loss