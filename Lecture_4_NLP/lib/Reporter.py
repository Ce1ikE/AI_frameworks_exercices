from pathlib import Path
from unicodedata import name
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from typing import Any
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
import time
import seaborn as sns

class Reporter:
    base_path = Path('./results')
    
    def __init__(
        self,
        save_path: str | Path = None,
        fontdict=None             
    ):
        self.root_path = None
        self.save_path = None
        self.fontdict = fontdict or {
            "fontsize": 10,
            "fontweight": "bold",
            "fontfamily": "monospace",
        }
        if save_path is not None:
            self.set_save_path(save_path)

    def set_save_path(self, new_path: str | Path):
        """
        If called with a base name (e.g. 'NLP'), create a timestamped experiment root:
            ./results/NLP_20251013_143512/
        If called with a subpath (e.g. <root>/model_name), set save_path inside that experiment.
        """
        new_path = Path(new_path)
        if new_path is None:
            raise ValueError("Please provide a valid path to save the results.")
        if new_path.is_file():
            raise ValueError("Please provide a directory path, not a file path.")

        if self.root_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            root_name = f"{new_path.name}_{timestamp}"
            self.root_path = self.base_path / root_name
            self.root_path.mkdir(parents=True, exist_ok=True)
            print(f"Created root: {self.root_path}")
            self.save_path = self.root_path

        else:
            if not str(new_path).startswith(str(self.root_path)):
                # relative subfolder inside the root
                self.save_path = self.root_path / new_path.name
            else:
                self.save_path = new_path
            self.save_path.mkdir(parents=True, exist_ok=True)
            print(f"Set save path to: {self.save_path}")

        return self.save_path

    def plot_dataset(
        self, 
        df: pd.DataFrame, 
        dataset_name: str = 'full_dataset',
        title: str = 'Number of Tickets per Category',
        xlabel: str = 'Category',
        ylabel: str = 'Number of Tickets',
    ):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        category_counts = df['category'].value_counts()
        ax.barh(
            category_counts.index, 
            category_counts.values, 
            color='skyblue', 
            edgecolor='black', 
            alpha=0.7,
        )
        for index, value in enumerate(category_counts.values):
            plt.text(
                value, 
                index, 
                str(value), 
                fontdict=self.fontdict,
                va='center',
                ha='left',
                fontsize=9
            )

        plt.grid(
            visible=True, 
            which='both', 
            axis='both', 
            color='0.9', 
            linestyle='--', 
            linewidth=1,
            alpha=0.5
        )
        plt.title(title + f" ({dataset_name})", fontdict=self.fontdict)
        plt.xlabel(xlabel, fontdict=self.fontdict)
        plt.ylabel(ylabel, fontdict=self.fontdict)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.save_path / f'dataset_distribution_{dataset_name}.svg', format='svg')
        plt.close()

    def plot_cv_results(
        self,
        cv_results: dict,
        model_name: str,
    ):
        # creates a DataFrame from the CV results
        df = pd.DataFrame(cv_results)
        y_values = None
        x_values = None

        # --- find all parameter columns (those starting with 'param_')
        param_cols = [col for col in df.columns if col.startswith('param_')]
        if not param_cols:
            print(f"[{model_name}] No parameters found in cv_results_. Skipping plot.")
            return
        
        # --- Identify target metric
        metric = 'mean_test_score' if 'mean_test_score' in df.columns else df.columns[-1]
        metric = 'mean_test_score'
        if 'mean_train_score' in df.columns:
            df['overfit_gap'] = df['mean_train_score'] - df['mean_test_score']
        
        # --- 1 PARAMETER -- simple line plot
        if len(param_cols) == 1:
            param = param_cols[0]
            xlabel = param.replace('param_', '').replace('classifier__', '').replace('_', ' ').title()

            x_values = df[param].apply(lambda v: v if np.isscalar(v) else str(v))

            try:
                x_values = x_values.astype(float)
                order = np.argsort(x_values)
                x_values = x_values.iloc[order]
                y_values = df[metric].iloc[order]
                y_train_values = df['mean_train_score'].iloc[order] if 'mean_train_score' in df.columns else None
            except Exception:
                y_values = df[metric]
                y_train_values = df['mean_train_score'] if 'mean_train_score' in df.columns else None

            if y_train_values is not None:
                plt.figure(figsize=(9, 6))
                plt.plot(x_values, y_train_values, marker='o', label='Mean Train Score', linestyle='--', alpha=0.7)
                plt.plot(x_values, y_values, marker='o', label='Mean Test Score')
                plt.title(f"Grid Search CV Results ({model_name})", fontdict=self.fontdict)
                plt.xlabel(xlabel, fontdict=self.fontdict)
                plt.ylabel(metric.replace('_', ' ').title(), fontdict=self.fontdict)
                plt.xticks(rotation=45, ha='right')
                plt.legend()
                plt.grid(alpha=0.4)
                plt.tight_layout()

                save_file = self.save_path / f'cv_results_{model_name}_train_vs_test.svg'
                plt.savefig(save_file, format='svg')
                plt.close()
                print(f"Saved 1D CV plot (train vs test) for {model_name} -> {save_file}")
                return
            plt.figure(figsize=(9, 6))
            plt.plot(x_values, y_values, marker='o', label='Mean Test Score')
            # plt.fill_between(x_values, y_values - df[metric].std(), y_values + df[metric].std(), alpha=0.2)
            plt.title(f"Grid Search CV Results ({model_name})", fontdict=self.fontdict)
            plt.xlabel(xlabel, fontdict=self.fontdict)
            plt.ylabel(metric.replace('_', ' ').title(), fontdict=self.fontdict)
            plt.xticks(rotation=45, ha='right')
            plt.grid(alpha=0.4)
            plt.tight_layout()

            save_file = self.save_path / f'cv_results_{model_name}.svg'
            plt.savefig(save_file, format='svg')
            plt.close()
            print(f"Saved 1D CV plot for {model_name} → {save_file}")
            return
        
        # --- 2 PARAMETERS -- heatmap
        elif len(param_cols) == 2:
            param_x, param_y = param_cols
            xlabel = param_x.replace('param_', '').replace('classifier__', '').replace('_', ' ').title()
            ylabel = param_y.replace('param_', '').replace('classifier__', '').replace('_', ' ').title()

            df_pivot = df.copy()

            # convert complex params (tuples, lists) to strings for pivot compatibility
            for p in (param_x, param_y):
                df_pivot[p] = df_pivot[p].apply(lambda v: v if np.isscalar(v) else str(v))

            pivot_table = df_pivot.pivot_table(
                values=metric, index=param_y, columns=param_x, aggfunc='mean'
            )

            plt.figure(figsize=(9, 7))
            sns.heatmap(
                pivot_table,
                annot=True,
                fmt=".3f",
                cmap="YlGnBu",
                cbar_kws={'label': metric.replace('_', ' ').title()}
            )
            plt.title(f"Grid Search CV Heatmap ({model_name})", fontdict=self.fontdict)
            plt.xlabel(xlabel, fontdict=self.fontdict)
            plt.ylabel(ylabel, fontdict=self.fontdict)
            plt.tight_layout()

            save_file = self.save_path / f'cv_heatmap_{model_name}.svg'
            plt.savefig(save_file, format='svg')
            plt.close()
            print(f"Saved 2D CV heatmap for {model_name} → {save_file}")
            return

    def plot_feature_importances(
        self,
        model: Any,
        feature_names: np.ndarray,
        model_name: str,
    ):
        importances = None

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            # multi-class
            if coef.ndim > 1:
                importances = np.mean(np.abs(coef), axis=0)
            else:
                importances = np.abs(coef)
        else:
            print(f"[{model_name}] Skipping feature importance plot — model type has no importances (e.g. MLP, SVC).")
            return 
        
        if len(importances) != len(feature_names):
            print(f"[{model_name}] Feature importance mismatch: {len(importances)} importances vs {len(feature_names)} features.")
            return


        # creates a DataFrame for visualization
        df_importances = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False)
        df_top = df_importances.head(20)

        plt.figure(figsize=(10, 6))
        plt.barh(df_top['feature'], df_top['importance'], color='skyblue')
        plt.title(f"Top 20 Feature Importances ({model_name})", fontdict=self.fontdict)
        plt.xlabel('Importance', fontdict=self.fontdict)
        plt.ylabel('Feature', fontdict=self.fontdict)
        plt.grid(visible=True, which='both', axis='x', color='0.9', linestyle='--', linewidth=1, alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.save_path / f'feature_importances_{model_name}.svg', format='svg')
        plt.close()

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        model_name: str,
        classes: list[str],
        cmap=plt.cm.Blues,
    ):
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(f"Confusion Matrix ({model_name})", fontdict=self.fontdict)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, ha='right', rotation=45)
        plt.yticks(tick_marks, classes)

        # integer format
        fmt = 'd' 
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontdict=self.fontdict)

        plt.ylabel('True label', fontdict=self.fontdict)
        plt.xlabel('Predicted label', fontdict=self.fontdict)
        plt.tight_layout()
        plt.savefig(self.save_path / f'confusion_matrix_{model_name}.svg', format='svg')
        plt.close()

    def open_subplot_figure(self, nbr_rows: int = 1, nbr_cols: int = 1):
        self.subplot_fig, self.subplot_axs = plt.subplots(nbr_rows, nbr_cols, figsize=(nbr_cols * 8, nbr_rows * 5))
        # ensure it's always a 1D array
        self.subplot_axs = np.array(self.subplot_axs).reshape(-1)  
        self.current_subplot_index = 0

    def plot_confusion_matrix__as_subplots(
        self,
        cm: np.ndarray,
        model_name: str,
        classes: list[str],
        cmap=plt.cm.Blues,
    ):
        # extend the array if needed
        self.subplot_axs = np.append(self.subplot_axs, [self.subplot_axs[-1]])  
        ax: plt.Axes = self.subplot_axs[self.current_subplot_index]

        ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.set_title(f"Confusion Matrix ({model_name})", fontdict=self.fontdict)
        plt.colorbar(ax.images[0], ax=ax)
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, ha='right', rotation=45)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, fontdict=self.fontdict)
        
        # integer
        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            ax.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontdict=self.fontdict)

        ax.set_ylabel('True label', fontdict=self.fontdict)
        ax.set_xlabel('Predicted label', fontdict=self.fontdict)
        self.current_subplot_index += 1

    def close_subplot_figure(self, filename: str):
        plt.tight_layout()
        plt.savefig(self.save_path / filename, format='svg')
        plt.close()

    def plot_performances(
        self,
        performance_dict: dict,
        metric: str = 'accuracy',
        title: str = 'Model Performance',
        xlabel: str = 'Models',
        ylabel: str = 'Performance',
        save_name: str = 'model_performance.svg'
    ):
        models = list(performance_dict.keys())
        scores = [performance_dict[model][metric] for model in models]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, scores, color='skyblue', edgecolor='black', alpha=0.7)
        plt.ylim(0, 1)  # assuming metric is between 0 and 1
        plt.title(title, fontdict=self.fontdict)
        plt.xlabel(xlabel, fontdict=self.fontdict)
        plt.ylabel(ylabel, fontdict=self.fontdict)
        plt.xticks(rotation=45, ha='right')

        for bar in bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2, 
                yval + 0.01, 
                f"{yval:.2f}", 
                ha='center', 
                va='bottom',
                fontdict=self.fontdict
            )

        plt.grid(
            visible=True, 
            which='both', 
            axis='y', 
            color='0.9', 
            linestyle='--', 
            linewidth=1,
            alpha=0.5
        )
        plt.tight_layout()
        plt.savefig(self.save_path / save_name, format='svg')
        plt.close()

    def save_performances(
        self,
        performance_dict: dict,
        filename: str = 'model_performance.json'
    ):
        if not filename.endswith('.json'):
            filename += '.json'
        with open(self.save_path / filename, 'w') as f:
            json.dump(performance_dict, f, indent=4)