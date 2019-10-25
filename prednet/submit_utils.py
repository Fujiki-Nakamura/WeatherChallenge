eval_i, eval_j = (130, 40)
eval_h, eval_w = 420, 340


def crop_eval_area(data):
    return data[:, :, eval_j:eval_j+eval_h, eval_i:eval_i+eval_w]
