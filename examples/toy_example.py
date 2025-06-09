import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.train import train_transformer
from src.analysis import generate_smiles


def main():
    """Run a minimal training and inference example."""

    # Paths for training data and where to save checkpoints/results
    training_csv = os.path.join('dataset', 'toy_dataset.csv')
    checkpoint_dir = os.path.join('model_checkpoints', 'toy_example')
    results_path = os.path.join('results', 'toy_example_samples.csv')

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # Train a tiny model for one epoch using classical attention
    train_transformer(
        training_data=training_csv,
        checkpoint_dir=checkpoint_dir,
        learning_rate=0.005,
        weight_decay=0.1,
        batch_size=4,
        epochs=1,
        validation_split=0.2,
        attn_type="classical",
        num_qubits=6,
        ansatz_layers=1,
        conditional_training=True,
        quantum_gradient_method="spsa",
        spsa_epsilon=0.01,
        sample_percentage=1.0,
        seed=42,
        classical_parameter_reduction=True,
        device="cpu",
        qpu_count=-1,
    )

    # Generate a few SMILES strings from the trained model
    checkpoint_path = os.path.join(checkpoint_dir, "model_epoch_1.pt")
    generate_smiles(
        checkpoint_path=checkpoint_path,
        save_dir=results_path,
        choose_best_val_epoch=False,
        num_of_model_queries=10,
        sampling_batch_size=5,
        imputation_dataset_path=training_csv,
        dataset_novelty_check_path=training_csv,
        device="cpu",
    )


if __name__ == "__main__":
    main()
