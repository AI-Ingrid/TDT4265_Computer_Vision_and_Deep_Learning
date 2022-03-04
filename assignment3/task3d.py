import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer


class TaskTwoModel(nn.Module):
    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()

        # Task 2a - Initialize the neural network
        num_filters = [32, 64, 128]  # Set number of filters in first conv layer
        self.num_classes = num_classes

        # Defining the neural network
        self.feature_extractor = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters[0],
                kernel_size=5,
                stride=1,
                padding=2
            ),

            # First max pool
            nn.MaxPool2d(stride=2, kernel_size=2),

            # Second convolutional layer
            nn.Conv2d(
                in_channels=num_filters[0],
                out_channels=num_filters[1],
                kernel_size=5,
                stride=1,
                padding=2
            ),

            # Second max pool
            nn.MaxPool2d(stride=2, kernel_size=2),

            # Third convolutional layer
            nn.Conv2d(
                in_channels=num_filters[1],
                out_channels=num_filters[2],
                kernel_size=5,
                stride=1,
                padding=2
            ),

            # Third max pool
            nn.MaxPool2d(stride=2, kernel_size=2),

        )
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 32*32*32
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Linear(4*4*128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # TODO: Implement this function (Task  2a)
        batch_size = x.shape[0]

        x = self.feature_extractor(x)
        # Flatten
        x = x.view(batch_size, -1)
        x = self.classifier(x)

        out = x

        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


class TaskThreeModel2(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()

        # Task 2a - Initialize the neural network
        num_filters = [32, 64, 128, 256]  # Set number of filters in first conv layer
        self.num_classes = num_classes

        # Defining the neural network
        self.feature_extractor = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters[0],
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(num_filters[0]),
            nn.SiLU(), #TODO inplace=true
            # First max pool
            nn.MaxPool2d(stride=2, kernel_size=2),

            # Second convolutional layer
            nn.Conv2d(
                in_channels=num_filters[0],
                out_channels=num_filters[1],
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(num_filters[1]),
            nn.Hardswish(inplace=True), #TODO inplace=true

            # Second max pool
            nn.MaxPool2d(stride=2, kernel_size=2),

            # Third convolutional layer
            nn.Conv2d(
                in_channels=num_filters[1],
                out_channels=num_filters[2],
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(num_filters[2]),
            nn.Hardswish(inplace=True), #TODO inplace=true

            # Third max pool
            nn.MaxPool2d(stride=2, kernel_size=2),

            # Fourth convolutional layer
            nn.Conv2d(
                in_channels=num_filters[2],
                out_channels=num_filters[3],
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(num_filters[3]),
            nn.Hardswish(inplace=True), #TODO inplace=true

            # Fourth max pool
            nn.MaxPool2d(stride=2, kernel_size=2),


        )
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 32*32*32
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Linear(2*2*256, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # TODO: Implement this function (Task  2a)
        batch_size = x.shape[0]

        x = self.feature_extractor(x)
        # Flatten
        x = x.view(batch_size, -1)
        x = self.classifier(x)

        out = x

        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


def create_plots(trainer_1: Trainer, trainer_2: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer_1.train_history["loss"], label="Training loss Model 1", npoints_to_average=10)
    utils.plot_loss(trainer_1.validation_history["loss"], label="Validation loss Model 1")
    utils.plot_loss(trainer_2.train_history["loss"], label="Training loss Model 2 (Improved)", npoints_to_average=10)
    utils.plot_loss(trainer_2.validation_history["loss"], label="Validation loss Model 2 (Improved)")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer_1.validation_history["accuracy"], label="Validation Accuracy Model 1")
    utils.plot_loss(trainer_2.validation_history["accuracy"], label="Validation Accuracy Model 2")
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

    # Model 1
    model_1 = TaskTwoModel(image_channels=3, num_classes=10)
    trainer_1 = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model_1,
        dataloaders
    )
    trainer_1.train()

    # Model 2
    model_2 = TaskTwoModel(image_channels=3, num_classes=10)
    trainer_2 = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model_2,
        dataloaders
    )
    trainer_2.train()

    # Plotting
    create_plots(trainer_1, trainer_2, "Improved and not improved models")

if __name__ == "__main__":
    main()