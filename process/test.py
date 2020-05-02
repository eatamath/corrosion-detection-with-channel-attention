
def test1():
    #for test
    hub_model = torch.hub.load(
        'moskomule/senet.pytorch',
        'se_resnet50',
        num_classes=NUM_CLASS,
    #     pretrained = True,
    )
    net = hub_model


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=P_lr)

    #
    model_weight = './pretrained_model/seresnet50-60a8950a85b2b.pkl'
    pretrained_dict=torch.load(model_weight)
    model_dict=net.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k != 'fc.weight' and k != 'fc.bias'}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

    checkpoint(net, optimizer, 0)
    return



def test2():
    #for test
    hub_model = torch.hub.load(
                'moskomule/senet.pytorch',
                'se_resnet50',
                num_classes=NUM_CLASS,
            )

    #### load model
    try:
        hub_model, epo = modelrestore(hub_model)
        print('Model successfully loaded')
        print('-' * 60)
    except Exception as e:
        print('Model not found, use the initial model')
        epo = 0
        print('-' * 60)

    net = hub_model.cuda()
    
    return