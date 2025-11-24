
import torch
import torch.nn as nn
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from pathlib import Path

from .global_constants import FONTDICT
from .model import Vocabulary

class NumpySerializer(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        
        return super().default(obj)

class Reporter:
    # NOTE: results_dir is expected to be created beforehand
    # and should be passed as argument to each method. such that functions can be used independently.
    
    @classmethod
    def save__training_results(
        cls,
        rnn: nn.Module,
        all_losses: list,
        results_dir: Path,
        learning_rate: float,
        n_hidden: int,
        n_epochs: int,
        n_letters: int,
        end: str
    ):
        all_losses_df = pd.DataFrame(
            columns=["iter","losses"],
            data=all_losses
        )
        all_losses_df.to_parquet(results_dir / "losses.parquet")
        
        torch.save(rnn.state_dict(),results_dir / "model_state")
        with open(results_dir / "settings.json",mode="w") as f:
            f.write(
                json.dumps(
                    cls=NumpySerializer,
                    obj={
                        "learning_rate" : learning_rate,
                        "n_hidden" : n_hidden,
                        "n_iters" : n_epochs,
                        "n_letters" : n_letters,
                        "total_training_time" : end,
                    },
                    indent=4
                )
            )

    @classmethod
    def save__training_results_cnn(
        cls,
        model: nn.Module,
        results_dir: Path,
        n_epochs: int,
        end: str
    ):
        torch.save(model.state_dict(),results_dir / "model_state")
        with open(results_dir / "settings.json",mode="w") as f:
            f.write(
                json.dumps(
                    cls=NumpySerializer,
                    obj={
                        "n_epochs" : n_epochs,
                        "total_training_time" : end,
                    },
                    indent=4
                )
            )

    @classmethod
    def save__lstm_results(
        cls,
        model: nn.Module,
        results_dir: Path,
        all_losses: list,
        n_epochs: int,
        learning_rate: float,
        embed_dim: int,
        hidden_size: int,
        batch_size: int,
        vocab_size: int,
        spacy_model: str,
        voc_creation_time: float,
        end: str
    ):
        all_losses_df = pd.DataFrame(
            columns=["iter","losses"],
            data=all_losses
        )
        all_losses_df.to_parquet(results_dir / "losses.parquet")
        
        torch.save(model.state_dict(),results_dir / "model_state")
        with open(results_dir / "settings.json",mode="w") as f:
            f.write(
                json.dumps(
                    cls=NumpySerializer,
                    obj={
                        "n_epochs" : n_epochs,
                        "learning_rate" : learning_rate,
                        "embed_dim" : embed_dim,
                        "hidden_size" : hidden_size,
                        "batch_size" : batch_size,
                        "vocab_size" : vocab_size,
                        "voc_creation_time" : voc_creation_time,
                        "spacy_model" : spacy_model,
                        "total_training_time" : end,
                    },
                    indent=4
                )
            )
 

    @classmethod
    def save_confusion_matrix(
        cls,
        confusion_matrix: pd.DataFrame,
        results_dir: Path
    ):
        confusion_matrix.to_parquet(results_dir / "confusion_matrix.parquet")

    @classmethod
    def save__dataset_splits(
        cls,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        results_dir: Path
    ):
        train_data.to_csv(results_dir / "train_dataset.csv")
        test_data.to_csv(results_dir / "test_dataset.csv")

        train_counts = train_data["category"].value_counts().to_dict()
        test_counts = test_data["category"].value_counts().to_dict()
        with open(results_dir / "train_dataset_category_counts.json",mode="w") as f:
            f.write(
                json.dumps(
                    cls=NumpySerializer,
                    obj=train_counts,
                    indent=4
                )
            )
        with open(results_dir / "test_dataset_category_counts.json",mode="w") as f:
            f.write(
                json.dumps(
                    cls=NumpySerializer,
                    obj=test_counts,
                    indent=4
                )
            )

    @classmethod
    def save__all_categories(
        cls,
        all_categories: list,
        results_dir: Path
    ):
        with open(results_dir / "all_categories.json",mode="w") as f:
            f.write(
                json.dumps(
                    cls=NumpySerializer,
                    obj=all_categories,
                    indent=4
                )
            )

    @classmethod
    def save__to_index_map(
        cls,
        to_index_map: dict,
        results_dir: Path
    ):
        """Saves the mapping from category to index as a JSON file."""
        with open(results_dir / "to_index_map.json",mode="w") as f:
            f.write(
                json.dumps(
                    cls=NumpySerializer,
                    obj=to_index_map,
                    indent=4
                )
            )

    @classmethod
    def save__all_letters(
        cls,
        all_letters: str,
        results_dir: Path
    ):
        with open(results_dir / "all_letters.txt",mode="w") as f:
            f.write(all_letters)

    @classmethod
    def save__vocabulary(
        cls,
        vocab: Vocabulary,
        results_dir: Path
    ):
        with open(results_dir / "vocab.json",mode="w") as f:
            f.write(
                json.dumps(
                    cls=NumpySerializer,
                    obj={
                        "word2idx" : vocab.word2idx,
                        "idx2word" : vocab.idx2word,
                        "min_freq" : vocab.min_freq
                    },
                    indent=4
                )
            )

    @classmethod
    def plot__datasets_distribution(
        cls,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        results_dir: Path
    ):
        from matplotlib import pyplot as plt
        import seaborn as sns
        from .global_constants import FONTDICT, RESULTS

        plt.figure(figsize=(8,5))
        plt.title("Dataset Category Distribution",fontdict=FONTDICT)
        combined_df = pd.concat(
            [
                train_data.assign(dataset="train"),
                test_data.assign(dataset="test")
            ],
            ignore_index=True
        )
        sns.countplot(
            data=combined_df,
            x="category",
            hue="dataset",
            palette="Set2"
        )
        plt.xlabel("Category",fontdict=FONTDICT)
        plt.ylabel("Count",fontdict=FONTDICT)
        plt.xticks(rotation=45,ha="right")
        plt.grid(visible=True,which="both",alpha=0.8)
        plt.tight_layout()
        plt.savefig(results_dir / "dataset_languages_names_category_distribution.svg",format="svg")
        plt.close()

    @classmethod
    def plot__loss_over_time(
        cls,
        results_dir: Path
    ):

        losses_path = results_dir / "losses.parquet"
        losses_df = pd.read_parquet(losses_path)
        iters = losses_df["iter"].values
        losses = losses_df["losses"].values

        plt.figure(figsize=(8,5))
        plt.title("Training Loss",fontdict=FONTDICT)
        plt.plot(iters, losses,c="blue")
        plt.xlabel("iteration",fontdict=FONTDICT)
        plt.ylabel("loss",fontdict=FONTDICT)
        plt.grid(visible=True,which="both",alpha=0.8)
        plt.tight_layout()
        plt.savefig(results_dir / "iteration_over_loss.svg",format="svg")
        plt.close()

    @classmethod
    def plot__confusion_matrix(
        cls,
        results_dir: Path
    ):
        confusion_matrix_path = results_dir / "confusion_matrix.parquet"
        confusion_matrix_df = pd.read_parquet(confusion_matrix_path)

        plt.figure(figsize=(10,8))
        plt.title("Confusion Matrix",fontdict=FONTDICT)
        sns.heatmap(
            confusion_matrix_df,
            annot_kws=FONTDICT,
            annot=True,
            fmt="d",
            # https://matplotlib.org/stable/users/explain/colors/colormaps.html
            cmap=plt.cm.get_cmap("plasma"),
            linewidths=.5,
            linecolor='black',
            xticklabels=confusion_matrix_df.columns,
            yticklabels=confusion_matrix_df.index
        )
        plt.xlabel("Predicted Category",fontdict=FONTDICT)
        plt.ylabel("True Category",fontdict=FONTDICT)
        plt.tight_layout()
        plt.savefig(results_dir / "confusion_matrix.svg",format="svg")
        plt.close()

        # normalize the confusion matrix
        confusion_matrix_df = confusion_matrix_df.div(
            confusion_matrix_df.sum(axis=1),
            axis=0
        )

        plt.figure(figsize=(10,8))
        plt.title("Confusion Matrix",fontdict=FONTDICT)
        sns.heatmap(
            confusion_matrix_df,
            annot_kws=FONTDICT,
            annot=True,
            fmt=".2f",
            vmax=1.0,
            vmin=0.0,
            # https://matplotlib.org/stable/users/explain/colors/colormaps.html
            cmap=plt.cm.get_cmap("plasma"),
            linewidths=.5,
            linecolor='black',
            xticklabels=confusion_matrix_df.columns,
            yticklabels=confusion_matrix_df.index
        )
        plt.xlabel("Predicted Category",fontdict=FONTDICT)
        plt.ylabel("True Category",fontdict=FONTDICT)
        plt.tight_layout()
        plt.savefig(results_dir / "confusion_matrix_normalized.svg",format="svg")
        plt.close()

    @classmethod   
    def save__cnn_fashion_mnist_results(
        cls,
        training_losses_df: pd.DataFrame,
        validation_losses_df: pd.DataFrame,
        confusion_matrix: pd.DataFrame,
        results_dir: Path
    ):
        training_losses_df.to_parquet(results_dir / "training_losses.parquet")
        validation_losses_df.to_parquet(results_dir / "validation_losses.parquet") 
        confusion_matrix.to_parquet(results_dir / "confusion_matrix.parquet")

    @classmethod    
    def plot__cnn_fashion_mnist_results(
        cls,
        results_dir: Path,
        confusion_matrix: pd.DataFrame
    ):
        plt.figure(figsize=(8, 5))
        training_losses_df = pd.read_parquet(results_dir / "training_losses.parquet")
        validation_losses_df = pd.read_parquet(results_dir / "validation_losses.parquet")
        plt.plot(
            training_losses_df["step"],
            training_losses_df["losses"],
            label='Training Loss'
        )
        plt.plot(
            validation_losses_df["step"],
            validation_losses_df["losses"],
            label='Validation Loss'
        )
        plt.xlabel("Step", fontdict=FONTDICT)
        plt.ylabel("Loss", fontdict=FONTDICT)
        plt.title("Training and Validation Loss", fontdict=FONTDICT)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(results_dir / "loss_plot.svg", format="svg")
        plt.close()

        # --------------- confusion matrix --------------- #
        plt.figure(figsize=(10,8))
        plt.title("Confusion Matrix",fontdict=FONTDICT)
        sns.heatmap(
            confusion_matrix,
            annot_kws=FONTDICT,
            annot=True,
            fmt="d",
            # https://matplotlib.org/stable/users/explain/colors/colormaps.html
            cmap=plt.cm.get_cmap("plasma"),
            linewidths=.5,
            linecolor='black',
            xticklabels=confusion_matrix.columns,
            yticklabels=confusion_matrix.index
        )
        plt.xlabel("Predicted Category",fontdict=FONTDICT)
        plt.ylabel("True Category",fontdict=FONTDICT)
        plt.tight_layout()
        plt.savefig(results_dir / "confusion_matrix.svg",format="svg")


    @classmethod
    def save__cnn_beexants_results(
        cls,
        training_losses_df: pd.DataFrame,
        validation_losses_df: pd.DataFrame,
        confusion_matrix: pd.DataFrame,
        accuracy_training_df: pd.DataFrame,
        accuracy_validation_df: pd.DataFrame,
        best_model_wts: dict,
        n_epochs: int,
        learning_rate: float,
        momentum: float,
        end: str,
        results_dir: Path
    ):
        training_losses_df.to_parquet(results_dir / "training_losses.parquet")
        validation_losses_df.to_parquet(results_dir / "validation_losses.parquet") 
        accuracy_training_df.to_parquet(results_dir / "accuracy_training.parquet")
        accuracy_validation_df.to_parquet(results_dir / "accuracy_validation.parquet")
        confusion_matrix.to_parquet(results_dir / "confusion_matrix.parquet")

        torch.save(best_model_wts, results_dir / "best_model_state.pth")

        with open(results_dir / "settings.json",mode="w") as f:
            f.write(
                json.dumps(
                    cls=NumpySerializer,
                    obj={
                        "n_epochs" : n_epochs,
                        "learning_rate" : learning_rate,
                        "momentum" : momentum,
                        "total_training_time" : end,
                    },
                    indent=4
                )
            )

    @classmethod    
    def plot__cnn_beexants_results(
        cls,
        results_dir: Path,
        confusion_matrix: pd.DataFrame
    ):
        plt.figure(figsize=(8, 5))
        training_losses_df = pd.read_parquet(results_dir / "training_losses.parquet")
        validation_losses_df = pd.read_parquet(results_dir / "validation_losses.parquet")
        accuracy_training_df = pd.read_parquet(results_dir / "accuracy_training.parquet")
        accuracy_validation_df = pd.read_parquet(results_dir / "accuracy_validation.parquet")
        plt.plot(
            training_losses_df["epoch"],
            training_losses_df["losses"],
            label='Training Loss'
        )
        plt.plot(
            validation_losses_df["epoch"],
            validation_losses_df["losses"],
            label='Validation Loss'
        )
        plt.plot(
            accuracy_training_df["epoch"],
            accuracy_training_df["accuracy"],
            label='Training Accuracy'
        )
        plt.plot(
            accuracy_validation_df["epoch"],
            accuracy_validation_df["accuracy"],
            label='Validation Accuracy'
        )
        plt.xlabel("Epoch", fontdict=FONTDICT)
        plt.ylabel("Loss", fontdict=FONTDICT)
        plt.title("Training and Validation | Loss and Accuracy", fontdict=FONTDICT)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(results_dir / "loss_plot.svg", format="svg")
        plt.close()

        # --------------- confusion matrix --------------- #
        plt.figure(figsize=(10,8))
        plt.title("Confusion Matrix",fontdict=FONTDICT)
        sns.heatmap(
            confusion_matrix,
            annot_kws=FONTDICT,
            annot=True,
            fmt="d",
            vmax=1.0,
            vmin=0.0,
            cmap=plt.cm.get_cmap("plasma"),
            linewidths=.5,
            linecolor='black',
            xticklabels=confusion_matrix.columns,
            yticklabels=confusion_matrix.index
        )
        plt.xlabel("Predicted Category",fontdict=FONTDICT)
        plt.ylabel("True Category",fontdict=FONTDICT)
        plt.tight_layout()
        plt.savefig(results_dir / "confusion_matrix.svg",format="svg")


    