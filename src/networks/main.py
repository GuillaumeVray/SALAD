from .resnet18 import ResNet18Autoencoder
from .unet import UNET

def build_network(net_name, rep_dim):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('CXR_resnet18', 'unet')
    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'CXR_resnet18':
        ae_net = ResNet18Autoencoder(rep_dim)

    if net_name == 'unet':
        ae_net = UNET(rep_dim)

    return ae_net
