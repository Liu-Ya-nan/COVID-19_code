from models import resnet, densenet, mobilenet, mobilenetv2, c3d, squeezenet,  shufflenet,  shufflenetv2, resnext, DenseNetModel

def generate_model(model_type='resnet', model_depth=18,
                   sample_size=256,
                   sample_duration=348, resnet_shortcut='B',
                   num_classes=2):
    assert model_type in ['resnet', 'densenet', 'mobilenet', 'mobilenetv2', 'c3d', 'squeezenet', 'shufflenet', 'shufflenetv2', 'resnext', 'DenseNetModel']


    if model_type == 'DenseNetModel':
        model = DenseNetModel.get_model(
            num_classes=num_classes,
            sample_size=sample_size,
            sample_duration=sample_duration)

    if model_type == 'c3d':
        model = c3d.get_model(
            num_classes=num_classes,
            sample_size=sample_size,
            sample_duration=sample_duration)
    elif model_type == 'squeezenet':
        model = squeezenet.get_model(
            version=1.1,
            num_classes=num_classes,
            sample_size=sample_size,
            sample_duration=sample_duration)
    elif model_type == 'shufflenet':
        model = shufflenet.get_model(
            groups=3,
            width_mult=1.0,
            num_classes=num_classes)
    elif model_type == 'shufflenetv2':
        model = shufflenetv2.get_model(
            num_classes=num_classes,
            sample_size=sample_size,
            width_mult=1.0)
    elif model_type == 'mobilenet':
        model = mobilenet.get_model(
            num_classes=num_classes,
            sample_size=sample_size,
            width_mult=1.0)
    elif model_type == 'mobilenetv2':
        model = mobilenetv2.get_model(
            num_classes=num_classes,
            sample_size=sample_size,
            width_mult=1.0)
    elif model_type == 'resnext':
        assert model_depth in [50, 101, 152]
        if model_depth == 50:
            model = resnext.resnext50(
                num_classes=num_classes,
                shortcut_type=resnet_shortcut,
                cardinality=32,
                sample_size=sample_size,
                sample_duration=sample_duration)
        elif model_depth == 101:
            model = resnext.resnext101(
                num_classes=num_classes,
                shortcut_type=resnet_shortcut,
                cardinality=32,
                sample_size=sample_size,
                sample_duration=sample_duration)
        elif model_depth == 152:
            model = resnext.resnext152(
                num_classes=num_classes,
                shortcut_type=resnet_shortcut,
                cardinality=32,
                sample_size=sample_size,
                sample_duration=sample_duration)

    elif model_type == 'densenet':
        assert model_depth in [101, 121]
        if model_depth == 121:
            model = densenet.generate_model(model_depth=121,
                                            num_classes=num_classes,
                                            # n_input_channels=opt.n_input_channels,
                                            conv1_t_size=7,
                                            conv1_t_stride=1,
                                            no_max_pool='store_true')
        elif model_depth == 101:
            model = densenet.generate_model(model_depth=101,
                                            num_classes=num_classes,
                                            # n_input_channels=opt.n_input_channels,
                                            conv1_t_size=7,
                                            conv1_t_stride=1,
                                            no_max_pool='store_true')

    elif model_type == 'resnet':
        assert model_depth in [10, 18, 34, 50, 101, 152, 200]
        if model_depth == 10:
            model = resnet.resnet10(
                num_classes=num_classes,
                shortcut_type=resnet_shortcut,
                sample_size=sample_size,
                sample_duration=sample_duration)
        elif model_depth == 18:
            model = resnet.resnet18(
                num_classes=num_classes,
                shortcut_type=resnet_shortcut,
                sample_size=sample_size,
                sample_duration=sample_duration)
        elif model_depth == 34:
            model = resnet.resnet34(
                num_classes=num_classes,
                shortcut_type=resnet_shortcut,
                sample_size=sample_size,
                sample_duration=sample_duration)
        elif model_depth == 50:
            model = resnet.resnet50(
                num_classes=num_classes,
                shortcut_type=resnet_shortcut,
                sample_size=sample_size,
                sample_duration=sample_duration)
        elif model_depth == 101:
            model = resnet.resnet101(
                num_classes=num_classes,
                shortcut_type=resnet_shortcut,
                sample_size=sample_size,
                sample_duration=sample_duration)
        elif model_depth == 152:
            model = resnet.resnet152(
                num_classes=num_classes,
                shortcut_type=resnet_shortcut,
                sample_size=sample_size,
                sample_duration=sample_duration)
        elif model_depth == 200:
            model = resnet.resnet200(
                num_classes=num_classes,
                shortcut_type=resnet_shortcut,
                sample_size=sample_size,
                sample_duration=sample_duration)

    return model
