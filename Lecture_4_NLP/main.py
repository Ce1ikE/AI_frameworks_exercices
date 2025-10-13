from pathlib import Path
import re
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import spacy
from spacy import displacy
import pandas as pd
import matplotlib.pyplot as plt

from lib.model_definitions import ModelStore
from lib.DataHandler import DataHandler
from lib.Reporter import Reporter

ORIGINAL_DATASET = './datasets/tickets.csv'
CLEANED_DATASET = './datasets/tickets_cleaned.csv'
TRAIN_DATASET = './datasets/tickets_train.csv'
TEST_DATASET = './datasets/tickets_test.csv'
SAVE_PATH = Path('NLP')
RANDOM_SEED = 42

reporter = Reporter(save_path=SAVE_PATH)

def nlp__intro_nlp():
    # for other languages, see https://spacy.io/usage/models
    nlp = spacy.load('en_core_web_sm')
    text = "'The humanitarian situation in Gaza was extremely dire before these hostilities; now it will only deteriorate exponentially,' Guterres told reporters at the UN in New York yesterday. 'Medical equipment, food, fuel and other humanitarian supplies are desperately needed, along with access for humanitarian personnel.'"
    doc = nlp(text)
    for sentence in doc.sents:
        print(sentence)
    for element in doc:
        print(element.text, element.lemma_, element.pos_)   

    for ent in doc.ents:
        print(ent.text, ent.label_)

    html_dep = displacy.render(doc, style = 'dep',page=True)
    html_dep_file = open("dep.html", "w", encoding="utf-8")
    html_dep_file.write(html_dep)
    html_dep_file.close()

    html_ent = displacy.render(doc, style = 'ent',page=True)
    html_ent_file = open("ent.html", "w", encoding="utf-8")
    html_ent_file.write(html_ent)
    html_ent_file.close()


def tickets__cleaning():
    df = pd.read_csv(ORIGINAL_DATASET)
    df = DataHandler.clean_data(df)
    df.to_csv(CLEANED_DATASET, index=False)
    print(f"Cleaned data saved to {CLEANED_DATASET}")

def tickets__saving():
    df = pd.read_csv(CLEANED_DATASET)
    train_data, test_data = DataHandler.split_data(df, test_size=0.2, random_state=RANDOM_SEED)
    train_data.to_csv(TRAIN_DATASET, index=False)
    test_data.to_csv(TEST_DATASET, index=False)
    print(f"Train and test datasets saved to {TRAIN_DATASET} and {TEST_DATASET}")

def tickets__data_visualization():
    df_test = pd.read_csv(TEST_DATASET)
    df_train = pd.read_csv(TRAIN_DATASET)
    df = pd.read_csv(CLEANED_DATASET)
    dataframes = {
        'train': df_train,
        'test': df_test,
        'full': df
    }
    for name, df in dataframes.items():
        reporter.plot_dataset(df, dataset_name=name+'_dataset')
    
def tickets__training():

    def summarize_params(params):
        clean = {}
        for k, v in params.items():
            if isinstance(v, (int, float, str, bool, type(None))):
                clean[k] = v
            elif hasattr(v, '__class__'):
                clean[k] = v.__class__.__name__
            else:
                clean[k] = str(v)
        return clean
    
    def get_classes(model):
        if hasattr(model, "classes_"):
            return model.classes_
        if hasattr(model, "named_steps"):
            return get_classes(model.named_steps.get("classifier"))
        if hasattr(model, "estimators_"):
            for _, est in model.estimators_:
                found = get_classes(est)
                if found is not None:
                    return found
        return None

    def get_base_estimator(model):
        """
        Recursively get the first estimator that has either
        `feature_importances_` or `coef_`, regardless of nesting
        (Pipeline, VotingClassifier, StackingClassifier, etc.)
        """
        # Direct model with importances or coefficients
        if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):
            return model
        # Pipelines
        if hasattr(model, "named_steps"):
            clf = model.named_steps.get("classifier")
            if clf is not None:
                return get_base_estimator(clf)
        # Ensembles (Voting, Stacking, Bagging)
        if hasattr(model, "estimators_"):
            for _, est in model.estimators_:
                result = get_base_estimator(est)
                if result is not None:
                    return result
        # Fallback — nothing found
        return None

    # https://spacy.io/usage/processing-pipelines
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    df_train = pd.read_csv(TRAIN_DATASET)
    df_test = pd.read_csv(TEST_DATASET)

    x_train = df_train['text'].tolist()
    x_test = df_test['text'].tolist()
    y_train = df_train['category'].tolist()
    y_test = df_test['category'].tolist()

    # steps:
    # -----
    # 1) create dictionary of models
    # 2) train each model
    #    2.1) Tokenization (with spacy) txt -> [tokens]
    #    2.2) Vectorization (with CountVectorizer or TfidfVectorizer) [tokens] -> [vector]
    #    2.3) Classification (with LogisticRegression, MultinomialNB, DecisionTreeClassifier) [vector] -> label
    # 4) evaluate each model
    # 5) save results (accuracy, confusion matrix)

    tokenizer_settings = {
        'library': 'spacy',
        'model': 'en_core_web_sm',
        'version': spacy.__version__,
        'custom_negation_handling': True,
        'custom_regex': {
            'url_pattern': r'https?://\S+|www\.\S+',
            'email_pattern': r'\S+@\S+\.\S+',
        },
        'custom_preprocessing': {
            'lowercase': True,
            'remove_stopwords': True,
            'remove_punctuation': True,
            'remove_short_tokens': 3,
            'keep_alpha': True,
            'replace_urls': 'URL',
            'replace_emails': '<EMAIL>',
            'use_lemmas': True,
        }
    }
    # this is just a helper function to use spacy for tokenization
    def spacy_tokenizer(text):
        # https://arxiv.org/html/2402.01035v2
        # https://www.datacamp.com/blog/what-is-tokenization
        text = text.lower()
        URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
        EMAIL_PATTERN = re.compile(r"\S+@\S+\.\S+")
        text = URL_PATTERN.sub(r" URL ", text)
        text = EMAIL_PATTERN.sub(r" <EMAIL> ", text)

        doc = nlp(text)
        tokens = []
        for token in doc:
            if token.is_stop or token.is_punct or token.is_space:
                continue
            if not token.is_alpha or len(token.text) < 3:
                continue

            lemma = token.lemma_.strip().lower()
            if lemma == '-pron-':
                continue
            tokens.append(lemma)

        def _handle_negations(tokens):
        # https://medium.com/@MansiKukreja/clinical-text-negation-handling-using-negspacy-and-scispacy-233ce69ab2ac
            negations = {"no", "not", "n't", "never", "none", "nobody", "nothing", "neither", "nowhere", "hardly", "scarcely", "barely"}
            negated_tokens = []
            negate = False
            for token in tokens:
                if token in negations:
                    negate = True
                    continue
                if negate:
                    token = f"NOT_{token}"
                    negate = False
                negated_tokens.append(token)
            return negated_tokens
        
        tokens = _handle_negations(tokens)

        return tokens

    models = ModelStore.get_simple_model_definitions(callback_tokenizer=spacy_tokenizer)

    # 2) train each model
    for pipeline_model_name, model in models.items():
        print(f"Training model: {pipeline_model_name}")
        model.fit(x_train, y_train)

    
    color_maps = ['Blues', 'Greens', 'Oranges', 'Purples', 'Reds', 'YlGnBu', 'GnBu', 'BuPu']
    new_save_path = reporter.set_save_path(SAVE_PATH)
    performances = {}
    reporter.open_subplot_figure(nbr_rows=len(models) // 2, nbr_cols=2)
    # 3) evaluate each model
    for pipeline_model_name, model in models.items():
        print(f"{pipeline_model_name} Evaluating model: {pipeline_model_name}")
        reporter.set_save_path(new_save_path / pipeline_model_name.replace(" ", "_").replace("+", "__"))
        
        model_name: str = getattr(model, 'name', 'model')
        if not model_name or model_name == 'model':
            model_name = pipeline_model_name.replace(" ", "_").replace("+", "__")
        
        if isinstance(model, GridSearchCV):
            if hasattr(model, 'best_params_'):
                print(f"{pipeline_model_name} Best Parameters: {model.best_params_}")
            if hasattr(model, 'best_score_'):
                print(f"{pipeline_model_name} Best Cross-Validation Score: {model.best_score_:.4f}")
            if hasattr(model, 'cv_results_'):
                print(f"{pipeline_model_name} CV Results: {model.cv_results_}")
                reporter.plot_cv_results(
                    cv_results=model.cv_results_,
                    model_name=model_name,
                )
            model = model.best_estimator_
        
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        print(f"{pipeline_model_name} Accuracy: {accuracy:.4f}")

        classes = get_classes(model)
        if classes is None:
            classes = sorted(set(y_test))

        base_estimator = get_base_estimator(model)
        if base_estimator is not None and (hasattr(base_estimator, 'feature_importances_') or hasattr(base_estimator, 'coef_')):
            print(f"{pipeline_model_name} Base Estimator: {base_estimator.__class__.__name__}")
            
            if hasattr(model, 'named_steps') and 'vectorizer' in model.named_steps:
                feature_names = model.named_steps['vectorizer'].get_feature_names_out()
            else:
                feature_names = [f"feature_{i}" for i in range(base_estimator.n_features_in_)]
            print(f"{pipeline_model_name} Feature Names: {feature_names[:10]}... (total {len(feature_names)})")
                        
            reporter.plot_feature_importances(
                model=base_estimator,
                feature_names=feature_names,
                model_name=model_name,
            )

        reporter.plot_confusion_matrix(cm=cm,model_name=model_name,classes=classes.tolist())
        # just to make it look nicer when all confusion matrices are together
        # cmap=color_maps[list(models.keys()).index(pipeline_model_name) % len(color_maps)],
        reporter.plot_confusion_matrix__as_subplots(cm=cm,model_name=model_name,classes=classes.tolist())
        performances[pipeline_model_name] = {
            'model': model_name,
            'accuracy': accuracy,
            'num_classes': len(classes),
            'model_settings': summarize_params(model.get_params(deep=True)),
        }


    reporter.set_save_path(new_save_path)
    reporter.close_subplot_figure(filename='confusion_matrices_all_models.svg')
    reporter.plot_performances(
        performances, 
        metric='accuracy', 
        title='Model Accuracies', 
        xlabel='Models', 
        ylabel='Accuracy', 
        save_name='model_accuracies.svg'
    )
    # add tokenizer info to performance report
    if spacy_tokenizer:
        performances['tokenizer'] = 'spacy_tokenizer'
        performances['tokenizer_settings'] = tokenizer_settings
    reporter.save_performances(performances, filename='model_accuracies.json')

def main():
    # nlp__intro_nlp()
    # tickets__cleaning()
    # tickets__saving()
    tickets__data_visualization()
    tickets__training()

if __name__ == "__main__":
    main()
