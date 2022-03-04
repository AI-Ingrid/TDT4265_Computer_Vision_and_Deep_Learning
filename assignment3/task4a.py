import torchvision 
from torch import nn 
import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy


class ResNet(nn.Module):
    def __init__(self, num_classes, image_channels):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512)
        
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
        for param in self.model.layer4.parameters():
            param.requires_grad = True 

        # Task 4a - Initialize the neural network
        self.num_classes = 10


        # Defining the neural network



    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]

        x = self.model(x)
        # Flatten
        x = x.view(batch_size, -1)
        x = self.classifier(x)

        out = x

        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}.png"))
    plt.show()


def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 5
    dataloaders = load_cifar10(batch_size)
    model = ResNet(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.train()
    create_plots(trainer, "2b")
    
    # (2b) Report for final training
    train, validation, test = dataloaders
    print("---- TRAINING ----")
    train_loss, train_acc = compute_loss_and_accuracy(train, model, nn.CrossEntropyLoss())
    print("---- VALIDATION ----")
    val_loss, val_acc = compute_loss_and_accuracy(validation, model, nn.CrossEntropyLoss())
    print("---- TEST ----")
    test_loss, test_acc = compute_loss_and_accuracy(test, model, nn.CrossEntropyLoss())

if __name__ == "__main__":
    main()