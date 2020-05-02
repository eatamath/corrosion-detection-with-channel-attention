
def net_freeze_layer(net,no_grad:list):
    for name, value in net.named_parameters():
        if name in no_grad:
            value.requires_grad = False
        else:
            value.requires_grad = True
    return

