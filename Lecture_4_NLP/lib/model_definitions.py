from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB , BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC , SVC
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV


class ModelStore:
    @staticmethod
    def get_simple_model_definitions(callback_tokenizer):
        # https://www.handsonmentor.com/post/natural-language-processing-text-vectorization-classification-and-n-gram-models-with-python
        # ////////////////////////////////////////////////////////////////////////////////////////
        # ////////////////////////////////////////////////////////////////////////////////////////
        # 1) definition of models
        return {
            # ////////////////////////////////////////////////////////////////////////////////////////
            # countvectorizer + classifier combinations //////////////////////////////////////////////
            "CountVectorizer + MultinomialNB": Pipeline(
                steps=[
                    ("vectorizer", CountVectorizer(
                        tokenizer=callback_tokenizer,
                        ngram_range=(1, 2),
                        max_features=5000
                    )),
                    ("classifier", MultinomialNB())
                ]
            ),
            "CountVectorizer + BernoulliNB": Pipeline(
                steps=[
                    ("vectorizer", CountVectorizer(
                        tokenizer=callback_tokenizer,
                        ngram_range=(1, 2),
                        max_features=5000
                    )),
                    ("classifier", BernoulliNB())
                ]
            ),
            "CountVectorizer + DecisionTreeClassifier": Pipeline(
                steps=[
                    ("vectorizer", CountVectorizer(
                        tokenizer=callback_tokenizer,
                        ngram_range=(1, 2),
                        max_features=5000
                    )),
                    ("classifier", DecisionTreeClassifier())
                ]
            ),
            "CountVectorizer + LogisticRegression": Pipeline(
                steps=[
                    ("vectorizer", CountVectorizer(
                        tokenizer=callback_tokenizer,
                        ngram_range=(1, 2),
                        max_features=5000
                    )),
                    ("classifier", LogisticRegression(max_iter=2000))
                ]
            ),
            "CountVectorizer + RandomForestClassifier": Pipeline(
                steps=[
                    ("vectorizer", CountVectorizer(
                        tokenizer=callback_tokenizer,
                        ngram_range=(1, 2),
                        max_features=5000
                    )),
                    ("classifier", RandomForestClassifier())
                ]
            ),
            "CountVectorizer + MLPClassifier": Pipeline(
                steps=[
                    ("vectorizer", CountVectorizer(
                        tokenizer=callback_tokenizer,
                        ngram_range=(1, 2),
                        max_features=5000
                    )),
                    ("classifier", MLPClassifier(max_iter=2000))
                ]
            ),
            # ////////////////////////////////////////////////////////////////////////////////////////
            # all TF_IDFVectorizer + classifier combinations /////////////////////////////////////////
            "TF_IDFVectorizer + BernoulliNB": Pipeline(
                steps=[
                    ("vectorizer", TfidfVectorizer(
                        tokenizer=callback_tokenizer,
                        ngram_range=(1, 2),
                        max_features=5000
                    )),
                    ("classifier", BernoulliNB())
                ]
            ),
            "TF_IDFVectorizer + MultinomialNB": Pipeline(
                steps=[
                    ("vectorizer", TfidfVectorizer(
                        tokenizer=callback_tokenizer,
                        ngram_range=(1, 2),
                        max_features=5000
                    )),
                    ("classifier", MultinomialNB())
                ]
            ),
            "TF_IDFVectorizer + DecisionTreeClassifier": Pipeline(
                steps=[
                    ("vectorizer", TfidfVectorizer(
                        tokenizer=callback_tokenizer,
                        ngram_range=(1, 2),
                        max_features=5000
                    )),
                    ("classifier", DecisionTreeClassifier())
                ]
            ),
            "TF_IDFVectorizer + LogisticRegression": Pipeline(
                steps=[
                    ("vectorizer", TfidfVectorizer(
                        tokenizer=callback_tokenizer,
                        ngram_range=(1, 2),
                        max_features=5000
                    )),
                    ("classifier", LogisticRegression(max_iter=2000))
                ]
            ),
            "TF_IDFVectorizer + RandomForestClassifier": Pipeline(
                steps=[
                    ("vectorizer", TfidfVectorizer(
                        tokenizer=callback_tokenizer,
                        ngram_range=(1, 2),
                        max_features=5000
                    )),
                    ("classifier", RandomForestClassifier())
                ]
            ),
            "TF_IDFVectorizer + MLPClassifier": Pipeline(
                steps=[
                    ("vectorizer", TfidfVectorizer(
                        tokenizer=callback_tokenizer,
                        ngram_range=(1, 2),
                        max_features=5000
                    )),
                    ("classifier", MLPClassifier(max_iter=2000))
                ]
            ),
        }
        # ////////////////////////////////////////////////////////////////////////////////////////
        # ////////////////////////////////////////////////////////////////////////////////////////

    @staticmethod
    def get_tuned_model_definitions(callback_tokenizer):
        return {
            # ////////////////////////////////////////////////////////////////////////////////////////
            # Hyperparameter tuning most promising models from previous step with GridSearchCV ///////
            "CountVectorizer + LogisticRegression (GridSearchCV)": GridSearchCV(
                estimator=Pipeline(
                    steps=[
                        ("vectorizer", CountVectorizer(
                            tokenizer=callback_tokenizer,
                            ngram_range=(1, 2),
                            max_features=5000
                        )),
                        ("classifier", LogisticRegression(max_iter=2000))
                    ]
                ),
                param_grid={
                    "classifier__C": [0.01, 0.1, 1.0, 10.0],
                    "vectorizer__max_features": [1000, 5000, 10000]
                },
                scoring="accuracy",
                cv=10,
                return_train_score=True
            ),
            "CountVectorizer + MultinomialNB (GridSearchCV)": GridSearchCV(
                estimator=Pipeline(
                    steps=[
                        ("vectorizer", CountVectorizer(
                            tokenizer=callback_tokenizer,
                            ngram_range=(1, 2),
                            max_features=5000
                        )),
                        ("classifier", MultinomialNB())
                    ]
                ),
                param_grid={
                    "classifier__alpha": [0.01, 0.1, 1.0, 10.0],
                    "vectorizer__max_features": [1000, 5000, 10000]
                },
                scoring="accuracy",
                cv=10,
                return_train_score=True
            ),
            "CountVectorizer + MLPClassifier (GridSearchCV)": GridSearchCV(
                estimator=Pipeline(
                    steps=[
                        ("vectorizer", CountVectorizer(
                            tokenizer=callback_tokenizer,
                            ngram_range=(1, 2),
                            max_features=5000
                        )),
                        ("classifier", MLPClassifier(max_iter=2000))
                    ]
                ),
                param_grid={
                    "classifier__hidden_layer_sizes": [(100,), (100, 100), (100, 50, 25)],
                    "vectorizer__max_features": [1000, 5000, 10000]
                },
                scoring="accuracy",
                cv=10,
                return_train_score=True
            ),
        }
        # ////////////////////////////////////////////////////////////////////////////////////////
        # ////////////////////////////////////////////////////////////////////////////////////////

    @staticmethod
    def get_tuned_model_definitions_version_2(callback_tokenizer):
        return {
            # ////////////////////////////////////////////////////////////////////////////////////////
            # Hyperparameter tuning most promising models from previous step with GridSearchCV ///////
            "TF_IDFVectorizer + LogisticRegression (GridSearchCV)": GridSearchCV(
                estimator=Pipeline(
                    steps=[
                        ("vectorizer", TfidfVectorizer(
                            tokenizer=callback_tokenizer,
                            ngram_range=(1, 2),
                            max_features=5000
                        )),
                        ("classifier", LogisticRegression(max_iter=5000, class_weight='balanced'))
                    ]
                ),
                param_grid={
                    "classifier__C": [0.01, 0.1, 1.0, 5.0, 10.0],
                    "classifier__penalty": ['l1', 'l2'],
                    "classifier__solver": ['liblinear', 'saga'],
                    "vectorizer__max_features": [1000, 5000, 10000]
                },
                scoring="accuracy",
                cv=10,
                return_train_score=True
            ),
            "CountVectorizer + MultinomialNB (GridSearchCV)": GridSearchCV(
                estimator=Pipeline(
                    steps=[
                        ("vectorizer", CountVectorizer(
                            tokenizer=callback_tokenizer,
                            ngram_range=(1, 2),
                            max_features=5000
                        )),
                        ("classifier", MultinomialNB())
                    ]
                ),
                param_grid={
                    "classifier__alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0],
                    "vectorizer__max_features": [1000, 5000, 10000]
                },
                scoring="accuracy",
                cv=10,
                return_train_score=True
            ),
            "TF_IDFVectorizer + MultinomialNB (GridSearchCV)": GridSearchCV(
                estimator=Pipeline(
                    steps=[
                        ("vectorizer", TfidfVectorizer(
                            tokenizer=callback_tokenizer,
                            ngram_range=(1, 2),
                            max_features=5000,
                            sublinear_tf=True
                        )),
                        ("classifier", MultinomialNB())
                    ]
                ),
                param_grid={
                    "classifier__alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0],
                    "vectorizer__max_features": [1000, 5000, 10000]
                },
                scoring="accuracy",
                cv=10,
                return_train_score=True
            ),
            "TF_IDFVectorizer + LinearSVC (GridSearchCV)": GridSearchCV(
                estimator=Pipeline(
                    steps=[
                        ("vectorizer", TfidfVectorizer(
                            tokenizer=callback_tokenizer,
                            ngram_range=(1, 2),
                            max_features=5000,
                            sublinear_tf=True
                        )),
                        ("classifier", LinearSVC(max_iter=5000, class_weight='balanced'))
                    ]
                ),
                param_grid={
                    "classifier__C": [0.01, 0.1, 1.0, 5.0, 10.0],
                    "classifier__penalty": ['l1', 'l2'],
                    "classifier__loss": ['hinge', 'squared_hinge'],
                    "vectorizer__max_features": [1000, 5000, 10000]
                },
                scoring="accuracy",
                cv=10,
                return_train_score=True
            ),
        }
        # ////////////////////////////////////////////////////////////////////////////////////////
        # ////////////////////////////////////////////////////////////////////////////////////////
    
    @staticmethod
    def get_tuned_ensemble_model_definitions(callback_tokenizer):
        tfidf_vectorizer = TfidfVectorizer(
            tokenizer=callback_tokenizer,
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True
        )
        count_vectorizer = CountVectorizer(
            tokenizer=callback_tokenizer,
            ngram_range=(1, 2),
            max_features=5000
        )

        lr_clf = LogisticRegression(
            max_iter=5000,
            class_weight='balanced',
            C=0.1,
            penalty='l2',
            solver='saga'
        )

        svc_clf = CalibratedClassifierCV(
            estimator=LinearSVC(
                max_iter=5000,
                class_weight='balanced',
                C=0.1,
                penalty='l2',
                loss='hinge'
            ),
            method='sigmoid',
            cv=3
        )
        
        mnb_clf = MultinomialNB(alpha=2.0)

        lr_pipe = Pipeline([
            ("vectorizer", tfidf_vectorizer),
            ("classifier", lr_clf)
        ])

        svc_pipe = Pipeline([
            ("vectorizer", tfidf_vectorizer),
            ("classifier", svc_clf)
        ])

        mnb_pipe = Pipeline([
            ("vectorizer", count_vectorizer),
            ("classifier", mnb_clf)
        ])
        
        return {
            # ////////////////////////////////////////////////////////////////////////////////////////
            # Ensemble models based on best models from previous step ////////////////////////////////
            "VotingClassifier (soft) of top 3 models": VotingClassifier(
                estimators=[
                    ("lr", lr_pipe),
                    ("svc", svc_pipe),
                    ("mnb", mnb_pipe),
                ],
                voting='soft'
            ),
            "StackingClassifier of top 3 models": StackingClassifier(
                estimators=[
                    ("lr", lr_pipe),
                    ("svc", svc_pipe),
                    ("mnb", mnb_pipe),
                ],
                final_estimator=LogisticRegression(
                    max_iter=5000,
                    class_weight='balanced',
                    solver='liblinear'
                ),
                stack_method='predict_proba',
                cv=5
            ),
            
    }
    # ////////////////////////////////////////////////////////////////////////////////////////
    # ////////////////////////////////////////////////////////////////////////////////////////

