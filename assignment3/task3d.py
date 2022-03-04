from task3_model1 import *
from task3_model2 import *

def create_plots(trainer_1: Trainer3, trainer_2: Trainer3, name: str):
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
    learning_rate = 7e-2
    early_stop_count = 5
    dataloaders = load_cifar10(batch_size)

    # Model 1
    model_1 = Model1(image_channels=3, num_classes=10)
    trainer_1 = Trainer3(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model_1,
        dataloaders
    )
    trainer_1.train()

    learning_rate = 6e-2
    early_stop_count = 3
    dataloaders = load_cifar10(batch_size)
    # Model 2
    model_2 = Model2(image_channels=3, num_classes=10)
    trainer_2 = Trainer3(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model_2,
        dataloaders
    )
    trainer_2.train()
    trainer_2.load_best_model()

    # Plotting
    create_plots(trainer_1, trainer_2, "3d")

if __name__ == "__main__":
    main()